"""
SAM3 业务逻辑层
负责会话管理、点框分组、格式转换、文件解析

视频分割设计:
- 提示与计算解耦: 提示记录(frame_prompts)与提交计算分离
- 实时模式(auto_predict=True): 加提示后立即提交并计算该帧
- 批量模式(auto_predict=False, 仅离线): 提示先记为 dirty, submit 时统一提交并传播
- 插帧: 以首次计算帧为锚点, 每 frame_stride 帧计算一个关键帧,
  提示帧自动成为额外关键帧(事件帧), 非关键帧复用最近的前序关键帧结果
- 删除分层: 未提交=服务层自清; 已提交微调=重提覆盖; 删除一帧的提示,但不删物体=remove_object_inputs;
  删除物体=remove_object; 全部重置=clear_objects
"""

import torch
import numpy as np
from PIL import Image
from typing import Optional, Dict, List, Set, Tuple, Any, Generator
from pydantic import BaseModel
from dataclasses import dataclass, field
import json
import time
import uuid
from pathlib import Path
import threading
import logging
import io
import struct

logger = logging.getLogger(__name__)

import sys
# sys.path.append(str(Path(__file__).parent.parent))
from ..compute.engine import SAM3ComputeEngine

# ============ 数据模型 ============
@dataclass
class PointGroup:
    """点组：一组点+标签对应一个物体"""
    group_id: int           # 组ID, 由前端分配, 在输入前确定
    points: List[Tuple[float, float]] = field(default_factory=list) # 默认为空列表
    labels: List[int] = field(default_factory=list)
    box: Optional[Tuple[float, float, float, float]] = None  # (x1, y1, x2, y2)
    
    def add_point(self, x: float, y: float, label: int):
        self.points.append((x, y))
        self.labels.append(label)
    
    def set_box(self, x1: float, y1: float, x2: float, y2: float):
        self.box = (x1, y1, x2, y2)

    def clear(self):
        self.points.clear()
        self.labels.clear()
        self.box = None
    

@dataclass
class ImageSession:
    """图像交互式会话"""
    session_id: str
    image: Image.Image
    point_groups: Dict[int, PointGroup] = field(default_factory=dict)
    image_embeddings: Optional[torch.Tensor] = None

    # 统一存储所有 mask 和 group_id 顺序
    masks: Optional[torch.Tensor] = None  # (num_objects, H, W)
    group_ids: List[int] = field(default_factory=list)  # 按顺序对应 masks 的每个物体

    created_at: float = field(default_factory=time.time)
    active: bool = True

    def get_or_create_group(self, group_id: int) -> PointGroup:
        # 无论有没有, 都要返回, 这个行为有点危险
        # 这个不危险, 本身就是get或者create
        if group_id not in self.point_groups:
            self.point_groups[group_id] = PointGroup(group_id=group_id)
        return self.point_groups[group_id]
    
    def classify_groups(self) -> Tuple[List[PointGroup], List[PointGroup], List[PointGroup]]:
        """
        将组分类为：
        - pure_point: 只有点，没有框
        - pure_box: 只有框，没有点
        - mixed: 既有框，又有点
        """
        pure_point = []
        pure_box = []
        mixed = []
        
        for group in self.point_groups.values():
            has_points = len(group.points) > 0
            has_box = group.box is not None
            
            # 不会把既有框, 又有点的组算到只有框, 只有点的组
            # 这是一种策略, 不一定是最好的
            if has_points and not has_box:
                pure_point.append(group)
            elif has_box and not has_points:
                pure_box.append(group)
            elif has_points and has_box:
                mixed.append(group)
        
        return pure_point, pure_box, mixed
    
    def _groups_to_tensor(self, groups: List[PointGroup]) -> Tuple[Optional[torch.FloatTensor], Optional[torch.LongTensor], Optional[torch.FloatTensor]]:
        """
        将同类型组列表转换为模型输入格式
        
        SAM3 支持不同物体不同点数，不需要补齐
        """
        num_objects = len(groups)
        if num_objects == 0:
            return None, None, None
        
        all_points = []
        all_labels = []
        all_boxes = []
        
        for group in groups:
            if len(group.points) > 0:
                group_points = [[float(x), float(y)] for x, y in group.points] # 现在还是单个的点组
                group_labels = [int(l) for l in group.labels]
                all_points.append(group_points) # 把单个的点组变为多个点组 [num_points, 2] -> [groups, num_points, 2]
                all_labels.append(group_labels) # 把单个的点标签变为多个点标签 [num_labels] -> [groups, num_labels]
            
            if group.box is not None:
                all_boxes.append([float(v) for v in group.box]) # [x1,y1,x2,y2] -> [num_objects, 4]
        
        click_points = torch.FloatTensor([all_points]) if all_points else None # [groups, num_points, 2] -> [batch, groups, num_points, 2]
        click_labels = torch.LongTensor([all_labels]) if all_labels else None  # [groups, num_labels]  -> [batch, groups, num_labels]
        input_boxes = torch.FloatTensor([all_boxes]) if all_boxes else None    # [groups, num_labels] -> [batch, groups, num_labels]
        
        return click_points, click_labels, input_boxes
    
@dataclass
class VideoSession:
    """
    视频交互式会话

    提示按帧组织: frame_prompts[frame_idx][group_id] = PointGroup
    实时模式(auto_predict)下记录后立即提交并计算, 批量模式下 submit 时统一提交计算
    """
    session_id: str
    video_session: Dict # {"session", "video_height", "video_width", "num_frames"} 计算层会话,由SAM3ComputeEngine主引擎返回
    original_size: Optional[Tuple[int, int]] = None
    is_streaming: bool = False
    frame_count: int = 0
    created_at: float = field(default_factory=time.time)
    active: bool = True

    # 交互模式
    auto_predict: bool = True  # True=实时提交, False=批量提交

    # 抽帧配置
    frame_stride: int = 1  # 1=每帧都处理, N=每N帧处理一次

    # 待提交的提示(批量模式)
    frame_prompts: Dict[int, Dict[int, PointGroup]] = field(default_factory=dict)  # frame_idx -> group_id -> PointGroup, 包含所有的提示, 经过提交和未提交的
    dirty_prompts: Set[Tuple[int, int]] = field(default_factory=set)  # 待提交到计算层的 (frame_idx, group_id)

    # ---- 计算状态 ----
    keyframe_results: Dict[int, Dict] = field(default_factory=dict)  # frame_idx -> {"masks": Tensor, "groups": List[int]}
    anchor_frame: Optional[int] = None         # 插帧网格锚点(首个计算帧)
    last_computed_frame: Optional[int] = None  # 跟踪前沿(客户端帧号), 如果用户请求已经算过的帧, 则返回缓存内容, 不然则进行计算
    submitted_groups: List[int] = field(default_factory=list)   # 镜像底层 obj_id 创建顺序
    group_first_frame: Dict[int, int] = field(default_factory=dict)  # group_id -> 首次提交帧 记录物体在哪一帧开始收到提示, 只是让每个 group 在“自己的起点之前”不返回结果

    # ---- 流式专用 ----
    received_frame_count: int = 0               # 已收到的客户端帧数
    pushed_frame_count: int = 0                 # 已推入底层 session 的帧数(session 帧号)
    last_frame: Optional[Image.Image] = None    # 最近收到的一帧(可能尚未推入)

    # 这里的客户端指的是WebSocket对端, 也就是前端 -> 桌面端/浏览器 
    stream_c2s: Dict[int, int] = field(default_factory=dict)  # 客户端帧号 -> session 帧号
    stream_s2c: Dict[int, int] = field(default_factory=dict)  # session 帧号 -> 客户端帧号

    # 传播取消信号, 中途取消一段长传播的线程安全信号
    # 当前端点取消后, 能够立即停下来, 不再算下一个关键帧
    cancel_event: threading.Event = field(default_factory=threading.Event)


# ============ SessionManager ============

class SessionManager:
    """
    统一管理所有会话生命周期
    把之前的session的生命周期管理直接提出来一个类
    """

    def __init__(self, max_age_seconds: float = 3600):
        self.max_age_seconds = max_age_seconds
        self.image_sessions: Dict[str, ImageSession] = {}
        self.video_sessions: Dict[str, VideoSession] = {}
        self._lock = threading.Lock()
        self._start_cleanup_timer()

    # 这个是后台用来定期清理线程的, 但是不知道怎么用, 以及是否鲁棒
    def _start_cleanup_timer(self):
        def cleanup_loop():
            while True:
                time.sleep(60)
                try:
                    self.cleanup_expired()
                except Exception as e:
                    logger.error(f"Cleanup error: {e}")
        
        thread = threading.Thread(target=cleanup_loop, daemon=True)
        thread.start()
    
    def register_image_session(self, image: Image.Image) -> str:
        session_id = str(uuid.uuid4())
        with self._lock:
            self.image_sessions[session_id] = ImageSession(
                session_id=session_id, image=image,
            )
        logger.info(f"注册图像会话: {session_id}")
        return session_id

    def get_image_session(self, session_id: str) -> Optional[ImageSession]:
        return self.image_sessions.get(session_id)

    def delete_image_session(self, session_id: str) -> bool:
        with self._lock: # 同一时刻，只有一个线程能拿到锁，进去执行；其他线程必须在外面排队等。
            if session_id in self.image_sessions:
                del self.image_sessions[session_id]
                return True
        return False
    
    def register_video_session(self, video_session: Dict, is_streaming: bool,
                               frame_stride: int = 1, auto_predict: bool = True) -> str:
        session_id = str(uuid.uuid4())
        with self._lock:
            self.video_sessions[session_id] = VideoSession(
                session_id=session_id, video_session=video_session,
                is_streaming=is_streaming,
                frame_stride=frame_stride, # 插帧步长, 1 表示逐帧计算掩码
                auto_predict=auto_predict, # 是否自动提交
                frame_count=video_session["num_frames"], # 或为冗余声明, 但是还是保留, 理论上应该从顶层拿到num_frames, 而不是从计算层
                original_size=((video_session["video_height"], video_session["video_width"])
                                if video_session["video_height"] is not None else None), # 作为冗余字段, 用起来方便
            )
        logger.info(f"注册视频会话: {session_id}, streaming={is_streaming}")
        return session_id
    
    def get_video_session(self, session_id: str) -> Optional[VideoSession]:
        return self.video_sessions.get(session_id)

    def delete_video_session(self, session_id: str) -> bool:
        with self._lock:
            if session_id in self.video_sessions:
                del self.video_sessions[session_id]
                return True
        return False
    
    def cleanup_expired(self):
        current_time = time.time()
        with self._lock:
            for sid in [s for s, v in self.image_sessions.items() if current_time - v.created_at > self.max_age_seconds]:
                del self.image_sessions[sid]
            for sid in [s for s, v in self.video_sessions.items() if current_time - v.created_at > self.max_age_seconds]:
                del self.video_sessions[sid]
    
    def get_stats(self) -> Dict:
        return {
            "image_sessions": len(self.image_sessions),
            "video_sessions": len(self.video_sessions),
        }


# TODO
# 关于Text的还没有做

class ImagePromptFile(BaseModel):
    """图像 Prompt 文件格式"""
    version: str = "1.0"
    type: str = "image"
    groups: List[Dict[str, Any]]


class VideoPromptFile(BaseModel):
    """视频 Prompt 文件格式"""
    version: str = "1.0"
    type: str = "video"
    frames: List[Dict[str, Any]]

# ============ 业务服务层 ============

class SAM3ServiceLayer:
    """
    SAM3 业务服务层
    理论上服务层也可以像计算层一样重构解耦
    """
    
    def __init__(self, model_path: str = "/root/workspace/modelRepo/SAM3",
                 device: str = "cuda:1",
                 enable_tracker: bool = True,
                 enable_video: bool = False,
                 enable_text:bool = False):
        self.compute_engine = SAM3ComputeEngine(
            model_path=model_path, device=device,
            enable_tracker=enable_tracker, enable_video=enable_video, enable_image=False # 这个是Engine在命名时的问题, 还没有改
        )
        self.session_manager = SessionManager() # 创建一个Manager管理Session

        # 串行化所有GPU计算(单卡, 图像/视频共享), 这个后面会做优化的, 可以多GPU并行, 以及负载均衡的工作
        self._compute_lock = threading.Lock()
    
    def set_model(self, model_type: str, enabled: bool):
        # 这个enabled的命名并不好, 应该是emmm, 另一个名字, enabled应该是一种状态
        self.compute_engine.set_model(model_type, enabled)
    
    def get_model_status(self) -> Dict:
        return self.compute_engine.get_model_status()
    
    # ******************

    # ========== 图像分割 ==========
    def create_image_session(self, image: Image.Image) -> str:
        return self.session_manager.register_image_session(image)
    
    def close_image_session(self, session_id: str) -> Dict:
        ok = self.session_manager.delete_image_session(session_id)
        if not ok:
            raise ValueError(f"会话 {session_id} 不存在")
        return {"success": True, "message": f"会话 {session_id} 已删除"}

    def add_point_to_group(self, session_id: str, group_id: int,
                           x: float, y: float, label: int) -> Dict:
        session = self.session_manager.get_image_session(session_id)
        if session is None:
            raise ValueError(f"会话 {session_id} 不存在")
        group = session.get_or_create_group(group_id)
        group.add_point(x, y, label)
        return {
            "success": True,
            "group_id": group_id,
            "num_points": len(group.points),
            "message": "点已添加，调用 predict 进行推理",
        }
    
    def add_box_to_group(self, session_id: str, group_id: int,
                         x1: float, y1: float, x2: float, y2: float) -> Dict:
        session = self.session_manager.get_image_session(session_id)
        if session is None:
            raise ValueError(f"会话 {session_id} 不存在")
        group = session.get_or_create_group(group_id)
        group.set_box(x1, y1, x2, y2)
        return {
            "success": True,
            "group_id": group_id,
            "has_box": True,
            "message": "框已添加，调用 predict 进行推理",
        }
    
    # 对于图片的清除, 在服务层直接实现, 因为图片没有会话状态 
    # 清理点和框, 但不删除组
    def clear_group(self, session_id: str, group_id: int) -> Dict:
        session = self.session_manager.get_image_session(session_id)
        if session is None:
            raise ValueError(f"会话 {session_id} 不存在")
        if group_id in session.point_groups:
            # 将点组clear, 我印象里推力refine, 就是要重传, 
            # refine只是把image_embedding存起来了,来节约时间
            # (节约在Encoder的计算, 具体可以节约多少没实际测过)
            session.point_groups[group_id].clear() 
        return {
            "success": True,
            "group_id": group_id,
            "message": "组已清空，调用 predict 进行推理",
        }
    
    def delete_image_point(self, session_id: str, group_id: int,
                           point_index: int = -1) -> Dict:
        """
        删除图像某组中的单个点(默认最后一个, 支持撤销式交互)
        图像模型无状态, 下次 predict 自然用剩余点重算, 无需额外通知
        """
        session = self.session_manager.get_image_session(session_id)
        if session is None:
            raise ValueError(f"会话 {session_id} 不存在")
        group = session.point_groups.get(group_id)
        if group is None or not group.points:
            raise ValueError(f"组 {group_id} 没有可删除的点")
        if not -len(group.points) <= point_index < len(group.points):
            # 这还能倒着删, 有点抽象了
            raise ValueError(f"point_index 越界: {point_index}, 共 {len(group.points)} 个点")
        del group.points[point_index]
        del group.labels[point_index]
        return {
            "success": True,
            "group_id": group_id,
            "num_points": len(group.points),
            "message": "点已删除，调用 predict 进行推理",
        }

    # 一个组只有一个框
    def clear_image_box(self, session_id: str, group_id: int) -> Dict:
        """清除图像某组中的框提示(点保留), 下次 predict 生效"""
        session = self.session_manager.get_image_session(session_id)
        if session is None:
            raise ValueError(f"会话 {session_id} 不存在")
        group = session.point_groups.get(group_id)
        if group is None or group.box is None:
            raise ValueError(f"组 {group_id} 没有框")
        group.box = None
        return {
            "success": True,
            "group_id": group_id,
            "has_box": False,
            "message": "框已清除，调用 predict 进行推理",
        }

    # 删除组
    def delete_group(self, session_id: str, group_id: int) -> Dict:
        session = self.session_manager.get_image_session(session_id)
        if session is None:
            raise ValueError(f"会话 {session_id} 不存在")
        if group_id in session.point_groups:
            del session.point_groups[group_id] 

        # 从 masks 中删除对应 group, 这段写的有点啰嗦
        if session.masks is not None and group_id in session.group_ids:
            idx = session.group_ids.index(group_id)
            mask_list = [session.masks[i] for i in range(session.masks.shape[0]) if i != idx]
            session.group_ids.pop(idx)
            session.masks = torch.stack(mask_list, dim=0) if mask_list else None

        # 如果所有组都删完了，清除 embeddings
        if not session.point_groups:
            session.image_embeddings = None
            session.masks = None
            session.group_ids = []
        
        return {
            "masks_tensor": session.masks,
            "num_objects": session.masks.shape[0] if session.masks is not None else 0,
            "group_ids": session.group_ids,
        }



    def _predict_image_session(self, session: ImageSession) -> Dict:
        """
        执行图像推理(核心方法)
        
        按类型分组推理，合并结果按 group_id 排序
        """
        pure_point, pure_box, mixed = session.classify_groups()
        
        if not pure_point and not pure_box and not mixed:
            raise ValueError("没有可用的提示")
        
        # 记录各类型 group_id 和对应的 mask
        type_results = []  # [(group_ids, masks_tensor), ...]
        
        # 纯点推理
        if pure_point:
            pts, lbls, _ = session._groups_to_tensor(pure_point)
            group_ids = [g.group_id for g in pure_point]
            
            if session.image_embeddings is None:
                # 第一次算结果
                result = self.compute_engine.predict_prompt(
                    image=session.image,
                    click_points=pts,
                    click_labels=lbls,
                )
                session.image_embeddings = result["image_embeddings"]
            else:
                # 对于Sam3TrackerModel是支持输入mask去refine的, 
                # 但是对于Sam3TrackerVideoModel,当你输入mask的时候, 会直接将你输入的mask作为结果
                prev_mask = self._extract_masks(session, group_ids)
                result = self.compute_engine.predict_prompt(
                    image=None,
                    click_points=pts,
                    click_labels=lbls,
                    input_masks=prev_mask,
                    image_embeddings=session.image_embeddings,
                    original_size=(session.image.size[1], session.image.size[0]),
                )
            
            # 将纯点组的group的结果存在results里
            type_results.append((group_ids, result["masks"])) 
    
        # 纯框推理
        if pure_box:
            _, _, boxes = session._groups_to_tensor(pure_box) # 只需要boxes
            group_ids = [g.group_id for g in pure_box]
            
            if session.image_embeddings is None:
                result = self.compute_engine.predict_prompt(
                    image=session.image,
                    input_boxes=boxes,
                )
                session.image_embeddings = result["image_embeddings"]
            else:
                prev_mask = self._extract_masks(session, group_ids)
                result = self.compute_engine.predict_prompt(
                    image=None,
                    input_boxes=boxes,
                    input_masks=prev_mask,
                    image_embeddings=session.image_embeddings,
                    original_size=(session.image.size[1], session.image.size[0]),
                )
            
            # 将纯框组的group的结果存在results里
            type_results.append((group_ids, result["masks"]))
        
        # 混合推理
        if mixed:
            pts, lbls, boxes = session._groups_to_tensor(mixed)
            group_ids = [g.group_id for g in mixed]
            
            if session.image_embeddings is None:
                result = self.compute_engine.predict_prompt(
                    image=session.image,
                    click_points=pts,
                    click_labels=lbls,
                    input_boxes=boxes,
                )
                session.image_embeddings = result["image_embeddings"]
            else:
                prev_mask = self._extract_masks(session, group_ids)
                result = self.compute_engine.predict_prompt(
                    image=None,
                    click_points=pts,
                    click_labels=lbls,
                    input_boxes=boxes,
                    input_masks=prev_mask,
                    image_embeddings=session.image_embeddings,
                    original_size=(session.image.size[1], session.image.size[0]),
                )
            
            # 将点框结合的group的结果存在results里
            type_results.append((group_ids, result["masks"]))
        
        # 按照group_id排序结果
        # type_results的的shape是
        # [
        #   (point_group_ids,point_results["masks"]),
        #   (box_group_ids,box_results["masks"]),
        #   (mix_group_ids, mix_results["masks"]),
        # ]
        all_items = []
        for group_ids, masks in type_results:
            for gid, mask in zip(group_ids, masks):
                all_items.append((gid, mask))
        
        all_items.sort(key=lambda x: x[0]) # all_items的shape是[(group_id, mask),.....]

        if all_items:
            session.group_ids = [gid for gid, _ in all_items]
            session.masks = torch.stack([mask for _, mask in all_items], dim=0)
        else:
            session.group_ids = []
            session.masks = None

        return {
            "masks_tensor": session.masks,
            "num_objects": session.masks.shape[0] if session.masks is not None else 0,
            "group_ids": session.group_ids,
        }
    
    def _extract_masks(self, session: ImageSession, group_ids: List[int]) -> Optional[torch.Tensor]:
        """从 session 中提取指定 group_ids 的 mask"""
        if session.masks is None or not session.group_ids:
            return None
        
        indices = []
        for gid in group_ids:
            # 这些gid有些有可能是非法的
            try:
                idx = session.group_ids.index(gid)
                indices.append(idx)
            except ValueError:
                # 即便没有也会return None
                return None
        
        return session.masks[indices]
    
    # 今天没弄完, 以下是明天的TODO
    # 需要把image_prompt和video_prompt文件格式的示例给到assets里面 done, 
    # 并且要把image_prompt和video_prompt的定义再api.md里说明白 done
    # 需要把load_prompt按照Kimi Code里给的代码改为load_image_prompt_file和load_video_prompt_file done
    # 补上ImagePrompt和VideoPromptFile的数据类型 done
    # 审查完最后的predit_image 经过设查后,将add_prompt与_predict_image解耦
    def load_image_prompt_file(self, session_id: str, file_data: Dict,
                                merge_mode: str = "append") -> Dict:
        """
        加载图像 prompt 文件
        
        merge_mode:
        - "append": 追加到现有组（点追加，框覆盖）
        - "replace": 覆盖整个组（清空后重新加载）
        - "skip": 跳过已存在的组
        """
        session = self.session_manager.get_image_session(session_id)
        if session is None:
            raise ValueError(f"会话 {session_id} 不存在")
        if merge_mode not in ("append", "replace", "skip"):
            raise ValueError(f"merge_mode 必须是 append/replace/skip 之一，当前: {merge_mode}")
        
        if file_data.get("type") != "image":
            raise ValueError(f"期望 type='image'，实际为 '{file_data.get('type')}'")
        
        loaded_groups = []
        skipped_groups = []

        for group_data in file_data.get("groups", []):
            group_id = group_data["group_id"]

            # skip 模式：已存在则跳过
            if merge_mode == "skip" and group_id in session.point_groups:
                skipped_groups.append(group_id)
                continue
            
            # replace 模式：清空现有组
            if merge_mode == "replace" and group_id in session.point_groups:
                session.point_groups[group_id].clear()
            
            group = session.get_or_create_group(group_id)
            
            points = group_data.get("points", [])
            labels = group_data.get("labels", [])
            
            if len(points) != len(labels):
                raise ValueError(f"组 {group_id}: points({len(points)}) 和 labels({len(labels)}) 长度不匹配")
            
            for pt, lbl in zip(points, labels):
                group.add_point(pt[0], pt[1], lbl)
            
            box = group_data.get("box")
            if box is not None:
                group.set_box(box[0], box[1], box[2], box[3])
            
            loaded_groups.append({
                "group_id": group_id,
                "num_points": len(points),
                "has_box": box is not None,
            })
        
        return {
            "success": True,
            "session_id": session_id,
            "merge_mode": merge_mode,
            "groups_loaded": loaded_groups,
            "groups_skipped": skipped_groups,
            "total_groups": len(session.point_groups),
            "message": f"Prompt 已加载（模式: {merge_mode}），调用 predict 进行推理",
        }
    
    def load_video_prompt_file(self, session_id: str, file_data: Dict,
                               merge_mode: str = "append") -> Dict:
        """
        加载视频 prompt 文件（批量导入多帧提示）

        merge_mode:
        这个业务逻辑和图像是一样的, 只是在不同的帧去做
        - "append": 追加到现有（同同组存在则叠加）
        - "replace": 覆盖同帧同组（清空该帧该组后重新加载）
        - "skip": 跳过已存在的同帧同组
        """
        session = self._get_video_session(session_id)
        if merge_mode not in ("append", "replace", "skip"):
            raise ValueError(f"merge_mode 必须是 append/replace/skip 之一，当前: {merge_mode}")
        
        if file_data.get("type") != "video":
            raise ValueError(f"期望 type='video'，实际为 '{file_data.get('type')}'")

        loaded_frames = set() # 用于记录哪些帧被导入了提示
        skipped_groups = [] # 记录哪些(帧号, 组号)因为merge_mode='skip'被跳过
        
        for frame_data in file_data.get("frames", []):
            frame_idx = frame_data["frame_idx"]
            frame_loaded = False    # 该帧是否有至少一个组真正被加载(全被 skip 则不算)

            for group_data in frame_data.get("groups", []):
                group_id = group_data["group_id"]

                existing = session.frame_prompts.get(frame_idx, {}).get(group_id) # 找出已经存在的组(这个是在服务层的,不是计算层)

                # skip 模式：已存在则跳过
                if merge_mode == "skip" and existing is not None:
                    skipped_groups.append({"frame_idx": frame_idx, "group_id": group_id})
                    continue

                # replace 模式：清空该帧该组(服务层清空 + 已提交过则同步清底层:
                # 将这组的输入与输出彻底删除, 即便这一帧的组变为了空, 也能解决)
                # 这个也比较危险吧, 直接用了计算层的操作
                if merge_mode == "replace" and existing is not None:
                    existing.clear()
                    obj_id = self._obj_id_of(session, group_id)
                    if obj_id is not None:
                        self.compute_engine.remove_video_object_inputs(
                            session.video_session, obj_id,
                            self._to_session_idx(session, frame_idx))

                group = self._record_prompt(session, group_id, frame_idx) # 这个是用来给服务层做记账的函数, 用来得到服务层已经记录过的group

                points = group_data.get("points", [])
                labels = group_data.get("labels", [])
                if len(points) != len(labels):
                    raise ValueError(
                        f"帧 {frame_idx} 组 {group_id}: points({len(points)}) 和 "
                        f"labels({len(labels)}) 长度不匹配")

                for (x, y), lbl in zip(points, labels):
                    group.add_point(float(x), float(y), int(lbl))

                box = group_data.get("box")
                if box is not None:
                    group.set_box(float(box[0]), float(box[1]),
                                  float(box[2]), float(box[3]))

                frame_loaded = True

            if frame_loaded:
                loaded_frames.add(frame_idx)

        # 提示批量变化后, 最早导入帧及之后的缓存作废(前向因果)
        if loaded_frames:
            self._invalidate_from(session, min(loaded_frames))

        return {
            "success": True,
            "session_id": session_id,
            "merge_mode": merge_mode,
            "loaded_frames": sorted(loaded_frames),
            "skipped_groups": skipped_groups,
            "total_prompt_frames": len(session.frame_prompts),
            "message": "Prompt 已加载，调用 submit_video_prompts 进行传播",
        }
    
    def predict_image(self, session_id: str) -> Dict:
        """
        触发图像推理(用于交互式会话)
        屏蔽掉细节, 供路由层调用
        在 load_prompt 或添加点/框后手动触发推理
        """
        session = self.session_manager.get_image_session(session_id)
        if session is None:
            raise ValueError(f"会话 {session_id} 不存在")
        return self._predict_image_session(session)
    
    def predict_image_once(self, image: Image.Image, groups: List[Dict]) -> Dict:
        """
        单次图像分割(无会话，直接推理)
        
        注意：此功能可能不够原子化，是否保留看后续整体使用。
        这个方法可能会取代 _predict_image, 或者被 _predict_image 取代。
        因为 _predict_image 现在也改为主动触发才会去分割。
        但这个函数比较偏先把提示点完，没有会话了，直接结束。
        """
        session = ImageSession(session_id="temp", image=image)
        for i, group_data in enumerate(groups):
            group = session.get_or_create_group(i)
            for pt, lbl in zip(group_data.get("points", []), group_data.get("labels", [])):
                group.add_point(pt[0], pt[1], lbl)
            if "box" in group_data:
                b = group_data["box"]
                group.set_box(b[0], b[1], b[2], b[3])
        return self._predict_image_session(session)
    
    # ========== 文本分割 ==========
    def predict_text(self, image: Image.Image, text_prompt: str,
                     confidence_threshold: float = 0.5) -> Dict:
        """
        文本提示分割
        
        TODO: 
        1. compute/engine.py 中 TextPromptEngine.predict() 需要完整实现
        2. 确认 Sam3Model / Sam3Processor 的 API 用法
        3. 统一返回格式为 torch.Tensor (num_objects, H, W)
        4. 添加输入校验
        5. 测试多物体文本分割
        """
        raise NotImplementedError(
            "文本提示分割 (predict_text) 尚未实现。\n"
            "需要完成的工作：\n"
            "1. compute/engine.py: TextPromptEngine.load() 确认模型加载\n"
            "2. compute/engine.py: TextPromptEngine.predict() 实现推理逻辑\n"
            "3. 确认返回格式: {'masks': torch.Tensor(N,H,W), 'scores': List[float], 'num_objects': int}\n"
            "4. 添加 _validate_text_prompt() 等输入校验\n"
            "5. 测试端到端流程"
        )
    
    # ========== 视频分割 ==========
    def _get_video_session(self, session_id: str) -> VideoSession:
        session = self.session_manager.get_video_session(session_id)
        if session is None:
            raise ValueError(f"视频会话 {session_id} 不存在")
        return session
    
    def create_video_session(self, video_frames: Optional[List[Image.Image]] = None,
                             frame_stride: int = 1,
                             auto_predict: bool = True) -> str:
        """
        创建视频会话
        离线: 传入全部帧; 流式: 传 None, 之后用 push_video_frame 逐帧推入
        帧统一存 CPU 内存, 计算时再由底层搬上显存, 避免长视频占满显存
        """
        compute_session = self.compute_engine.init_video_session(
            video_frames, video_storage_device="cpu",
        ) # 由engine创建的推理对象

        return self.session_manager.register_video_session(
            compute_session, video_frames is None,
            frame_stride=frame_stride, auto_predict=auto_predict,
        )

    # ---- group_id 与底层 obj_id 的映射(服务层管 id, 计算层不管) ----
    # 类方法, 不依赖实例状态, 将前端和服务层的对象编号映射到计算层的group_id
    @staticmethod
    def _obj_id_of(session: VideoSession, group_id: int) -> Optional[int]:
        """submitted_groups[obj_id] = group_id; 未提交过返回 None"""
        try:
            return session.submitted_groups.index(group_id)
        except ValueError:
            return None

    # obj_id 不是加提示时就分配，而是提交计算时才分配, 
    # 也就是第一次把某组的提示真正推给底层的前一刻才分配
    def _ensure_obj_id(self, session: VideoSession, group_id: int) -> int:
        obj_id = self._obj_id_of(session, group_id)
        if obj_id is None:
            session.submitted_groups.append(group_id)
            obj_id = len(session.submitted_groups) - 1
        return obj_id

    # 用于映射帧号
    @staticmethod
    def _to_session_idx(session: VideoSession, frame_idx: int) -> int:
        """客户端帧号 -> 底层 session 帧号(离线恒等, 流式查映射)"""
        if session.is_streaming:
            return session.stream_c2s[frame_idx]
        return frame_idx

    # ---- 提示提交与计算 ----

    def _flush_dirty_frame(self, session: VideoSession, frame_idx: int) ->None:
        """
        把某帧的脏提示推入计算层, 当往某帧加过提示, 
        之后再加提示, 是将结果重算, 而不是只算增量
        """
        for (f, gid) in [d for d in session.dirty_prompts if d[0] == frame_idx]: # d[0]为frame_idx, 并且一个帧可能对应多个组
            group = session.frame_prompts.get(f, {}).get(gid)
            session.dirty_prompts.discard((f, gid)) # 字典的删除方法, 如果不存在, 什么也不做, 不报错

            # 没有组 or 组内没有提示
            if group is None or (not group.points and group.box is None):
               continue 
            
            obj_id = self._ensure_obj_id(session, gid) # 为group设置obj_id
            pts = [list(p) for p in group.points] if group.points else None
            # 流式会话 init 时无帧, 字典里是 (None, None), 会把底层
            # video_height 覆盖成 None; original_size 在首帧推入时记录
            self.compute_engine.add_video_prompt(
                session.video_session,
                frame_idx=self._to_session_idx(session, f),
                obj_id=obj_id,
                click_points=[[pts]] if pts else None,
                click_labels=[[group.labels]] if pts else None,
                input_boxes=[[list(group.box)]] if group.box is not None else None,
                original_size=session.original_size,
            ) # 补丁4a, 为了解决流式会话的问题, 但这个应该要路由层来承担这个责任, 给original_size
            
    
    def _cache_result(self, session: VideoSession, frame_idx: int,
                      masks: Optional[torch.Tensor]) -> Dict:
        """缓存一帧的计算结果(掩码搬回 CPU, groups 记录行对齐)"""
        if session.anchor_frame is None:
            session.anchor_frame = frame_idx
        res = {
            "masks": masks.cpu() if masks is not None else None,
            "groups": list(session.submitted_groups),
        }
        session.keyframe_results[frame_idx] = res # 将keyframe的结果缓存
        session.last_computed_frame = (frame_idx if session.last_computed_frame is None
                                       else max(session.last_computed_frame, frame_idx)) # 如果帧为比当前帧靠后, 则将其作为最新的一帧
        
        return res
    
    def _compute_and_cache(self, session: VideoSession, frame_idx: int) -> Dict:
        """计算某个客户端帧并缓存(0 物体防护: 没物体不调用底层)"""
        if not session.submitted_groups:
            return self._cache_result(session, frame_idx, None) # 0物体防护, 不会调用底层的engine
        out = self.compute_engine.predict_video_frame(
            session.video_session, self._to_session_idx(session, frame_idx))
        return self._cache_result(session, frame_idx, out["masks"])

    def _visible_result(self, session: VideoSession, frame_idx: int, res: Dict) -> Dict:
        """过滤掉在物体第一次提示之前的帧出现的mask, 只有第一次提示之后的mask才能看到"""
        keep = [i for i, g in enumerate(res["groups"])
                if session.group_first_frame.get(g, 0) <= frame_idx] # frame_idx在group_first_frame之后的,保留
        masks = res["masks"]
        if masks is not None:
            masks = masks[keep] if keep else None
        return {"masks": masks, "groups": [res["groups"][i] for i in keep]}
    
    def _latest_cached_before(self, session: VideoSession, frame_idx: int) -> Optional[int]:
        # 找一个不大于frame_idx的, 最近的一个已缓存的关键帧
        # 用于插帧复用之前的掩码
        candidates = [f for f in session.keyframe_results if f <= frame_idx]
        return max(candidates) if candidates else None

    def _invalidate_from(self, session: VideoSession, frame_idx: int) -> None:
        """提示变化后丢弃 frame_idx 及之后的缓存(前向因果, 之前的仍有效)"""
        for f in [f for f in session.keyframe_results if f >= frame_idx]:
            del session.keyframe_results[f] # 当我删除idx的帧的时候, 我将idx这帧和之后处理过的帧的结果全部删除
        session.last_computed_frame = max(session.keyframe_results) if session.keyframe_results else None
        if session.last_computed_frame is None:
            session.anchor_frame = None

    def _remove_group_everywhere(self, session: VideoSession, group_id: int) -> None:
        """移除一个组的对应的物体的全部痕迹: 底层物体、提示记录、缓存中的对应行"""
        obj_id = self._obj_id_of(session, group_id)
        if obj_id is not None:
            self.compute_engine.remove_video_object(session.video_session, obj_id) # 底层删除掉这个物体
            session.submitted_groups.pop(obj_id)  # 底层已重索引, 列表同步前移
        for f in list(session.frame_prompts):
            session.frame_prompts[f].pop(group_id, None) 
            if not session.frame_prompts[f]:
                del session.frame_prompts[f]
        session.dirty_prompts = {d for d in session.dirty_prompts if d[1] != group_id} # 将未提交的group_id的物体的提示也删除
        session.group_first_frame.pop(group_id, None)
        # 缓存掩码的行与 obj_id 对齐, 删对应行即可(其他物体的追踪互不影响)
        for res in session.keyframe_results.values():
            if group_id in res["groups"]:
                idx = res["groups"].index(group_id) # 得到对应的obj_idx
                keep = [i for i in range(len(res["groups"])) if i != idx] # 要保留的obj
                res["masks"] = res["masks"][keep] if (res["masks"] is not None and keep) else None
                res["groups"] = [res["groups"][i] for i in keep]

    # ---- 加提示(记录与计算解耦) ----
    def _record_prompt(self, session: VideoSession, group_id: int, frame_idx: int) -> PointGroup:
        group = session.frame_prompts.setdefault(frame_idx, {}).get(group_id)
        if group is None:
            # group is None会出现在这帧上, 这个group第一次出现
            group = PointGroup(group_id=group_id)
            session.frame_prompts[frame_idx][group_id] = group
        session.group_first_frame[group_id] = min(
            session.group_first_frame.get(group_id, frame_idx), frame_idx) # 是否将该帧作为这个group的第一个关键帧
        session.dirty_prompts.add((frame_idx, group_id))
        return group

    @staticmethod
    def _check_stream_prompt_frame(session: VideoSession, frame_idx: int) -> None:
        if session.is_streaming and frame_idx != session.received_frame_count - 1:
            raise ValueError("流式模式只能给最新收到的帧加提示") # 检查该帧是否为最新一帧

        
    def _ensure_prompt_frame_pushed(self, session: VideoSession, frame_idx: int) -> None:
        """
        流式下提示帧若尚未推入底层(非网格帧), 先补推——提示帧必成关键帧
        这个函数本身的设计就不是给非网格帧用的
        危险函数(待后续优化), 已知三个问题:
        1. 双算: _push_streaming_frame 推帧时算一遍, flush 提示后 _compute_and_cache
           再算一遍, 第一遍被覆盖(优化见 engine.process_frame 的 TODO)
        2. 不均匀时间步: 补推使底层帧序列间隔不等(如 0,3,6,7), 记忆注意力的时间
           位置编码信号变脏, 运动剧烈时可能影响精度
        3. 并发窗口: 依赖 last_frame 就是 frame_idx 那一帧, 若路由层并发处理
           推帧与提示消息可能推错帧(顺序处理则无此问题)
        """
        if not session.is_streaming or frame_idx in session.stream_c2s:
            return  # 离线无需补推; 已推入底层的帧直接返回, 如果在session.stream_c2s, 则意味着该帧已经被推入底层
        if frame_idx != session.received_frame_count - 1:
            # 契约: 只能补推最新收到的帧(底层只能顺序接帧)
            raise ValueError(
                f"只能补推最新帧: frame_idx={frame_idx}, "
                f"最新={session.received_frame_count - 1}")
        self._push_streaming_frame(session, session.last_frame)
         
    def add_video_point(self, session_id: str, group_id: int,
                        x: float, y: float, label: int, frame_idx: int) -> Dict:
        session = self._get_video_session(session_id)
        self._check_stream_prompt_frame(session, frame_idx)
        group = self._record_prompt(session, group_id, frame_idx)
        group.add_point(x, y, label)
        if not session.auto_predict:
            # 该帧及之后的旧缓存不含新提示, 作废
            # (否则前沿不退, 增量 submit 跳过这段, 提示永远算不上)
            # 是需要回滚前沿，重新propagate的
            self._invalidate_from(session, frame_idx) # 标记前面的掩码缓存脏了
            return {"success": True, "computed": False,
                    "message": "提示已记录, 调用 submit 计算"}
        self._ensure_prompt_frame_pushed(session, frame_idx)
        self._flush_dirty_frame(session, frame_idx) # 将prompt传给计算层
        res = self._compute_and_cache(session, frame_idx) # 返回结果
        # 该帧已用新提示重算(所以不杀本帧), 之后帧的旧缓存作废, 前沿回退到该帧
        self._invalidate_from(session, frame_idx + 1)
        return {"success": True, "computed": True, "frame_idx": frame_idx,
                **self._visible_result(session, frame_idx, res)}
    
    def add_video_box(self, session_id: str, group_id: int,
                    x1: float, y1: float, x2: float, y2: float, frame_idx: int) -> Dict:
        session = self._get_video_session(session_id)
        self._check_stream_prompt_frame(session, frame_idx)
        group = self._record_prompt(session, group_id, frame_idx)
        group.set_box(x1, y1, x2, y2)
        if not session.auto_predict:
            return {"success": True, "computed": False,
                    "message": "提示已记录, 调用 submit 计算"}
        self._ensure_prompt_frame_pushed(session, frame_idx)
        self._flush_dirty_frame(session, frame_idx)
        self._invalidate_from(session, frame_idx) # 标记前面的掩码缓存脏了
        res = self._compute_and_cache(session, frame_idx)
        # 该帧已用新提示重算(所以不杀本帧), 之后帧的旧缓存作废, 前沿回退到该帧
        self._invalidate_from(session, frame_idx + 1)
        return {"success": True, "computed": True, "frame_idx": frame_idx,
                **self._visible_result(session, frame_idx, res)}

    # ---- 删提示(细粒度: 单点/框; 空组自动升级为删除物体) ----
    def delete_video_point(self, session_id: str, group_id: int,
                           frame_idx: int, point_index: int) -> Dict:
        session = self._get_video_session(session_id)
        group = session.frame_prompts.get(frame_idx, {}).get(group_id)
        if group is None or point_index >= len(group.points):
            raise ValueError("该点不存在")
        if not -len(group.points) <= point_index < len(group.points): # 和image的delete_video_point一样
            raise ValueError(f"point_index 越界: {point_index}, 共 {len(group.points)} 个点")
        group.points.pop(point_index)
        group.labels.pop(point_index)
        self._after_prompt_fine_removed(session, group_id, frame_idx)
        return {"success": True, "remaining_points": len(group.points)}
    
    def clear_video_box(self, session_id: str, group_id: int, frame_idx: int) -> Dict:
        session = self._get_video_session(session_id)
        group = session.frame_prompts.get(frame_idx, {}).get(group_id)
        if group is None or group.box is None:
            raise ValueError("该框不存在")
        group.box = None
        self._after_prompt_fine_removed(session, group_id, frame_idx)
        return {"success": True}

    def _after_prompt_fine_removed(self, session: VideoSession, group_id: int, frame_idx: int) -> None:
        """细粒度删除善后: 同步底层输入 -> 重算最早提示帧 -> 失效缓存/空组升级删除"""
        group = session.frame_prompts.get(frame_idx, {}).get(group_id)
        obj_id = self._obj_id_of(session, group_id)
        if group is not None and (group.points or group.box is not None):
            if obj_id is not None:
                session.dirty_prompts.add((frame_idx, group_id))  # 剩余提示重推覆盖
        else:
            # 该帧提示已空: 清掉底层该帧的输入和输出
            if obj_id is not None:
                # 如果对象还存在
                self.compute_engine.remove_video_object_inputs(
                    session.video_session, obj_id, self._to_session_idx(session, frame_idx)) # 清除掉底层该帧的输入
            session.dirty_prompts.discard((frame_idx, group_id)) # 将未提交的提示删除
            if frame_idx in session.frame_prompts:
                session.frame_prompts[frame_idx].pop(group_id, None) # 删除掉该帧上的空组
                if not session.frame_prompts[frame_idx]:
                    del session.frame_prompts[frame_idx]  # 如果该帧的上的所有组都删光了

        # 该组在所有帧上都没有提示了 -> 空组升级, 删除整个物体
        if not any(group_id in groups for groups in session.frame_prompts.values()):
            self._remove_group_everywhere(session, group_id)
        # 只有提示到达过底层, 缓存才可能因此变脏: frame_idx 及之后作废(前向因果)
        if obj_id is not None:
            self._invalidate_from(session, frame_idx)

    def clear_video_group(self, session_id: str, group_id: int) -> Dict:
        """删除一个物体(组)及其在全部帧上的提示"""
        session = self._get_video_session(session_id)
        self._remove_group_everywhere(session, group_id)
        remaining = sorted({g for groups in session.frame_prompts.values() for g in groups})
        return {"success": True, "group_id": group_id, "remaining_groups": remaining}

    # ---- 流式推帧 ----
    def push_video_frame(self, session_id: str, frame: Image.Image) -> Dict:
        """流式收帧: 网格帧推入底层并track, 非关键帧复用最近结果(插帧)"""
        session = self._get_video_session(session_id)
        if not session.is_streaming:
            raise ValueError("非流式会话, 离线视频请用 submit/get_video_frame_result")
        session.received_frame_count += 1
        session.last_frame = frame

        # 补丁4b 首帧记录视频尺寸(PIL size=(w,h), original_size=(h,w)), 供提示提交使用
        if session.original_size is None:
            session.original_size = (frame.size[1], frame.size[0])
            session.video_session["video_height"] = frame.size[1]
            session.video_session["video_width"] = frame.size[0]


        client_idx = session.received_frame_count - 1

        # 为网格帧
        if session.anchor_frame is None or (client_idx - session.anchor_frame) % session.frame_stride == 0:
            res = self._push_streaming_frame(session, frame) # push了之后会有新的cache和keyframe_result
            return {"frame_idx": client_idx, "keyframe": True,
                    **self._visible_result(session, client_idx, res)}

        prev = self._latest_cached_before(session, client_idx)
        res = session.keyframe_results[prev] if prev is not None else {"masks": None, "groups": []}
        return {"frame_idx": client_idx, "keyframe": False, "reused_from": prev,
                **self._visible_result(session, client_idx, res)}

    def _push_streaming_frame(self, session: VideoSession, frame: Image.Image) -> Dict:
        """把最近收到的一帧推入底层 session 并完成该帧跟踪"""
        client_idx = session.received_frame_count - 1
        s_idx = session.pushed_frame_count
        out = self.compute_engine.process_video_frame(session.video_session, frame)
        session.stream_c2s[client_idx] = s_idx # 前端映射到后端
        session.stream_s2c[s_idx] = client_idx # 后端映射到前端
        session.pushed_frame_count += 1
        return self._cache_result(session, client_idx,
                                  out["masks"] if out["num_objects"] > 0 else None)
    
    # ---- 取结果 / 提交计算 ----
    def get_video_frame_result(self, session_id: str, frame_idx: int,
                               compute_if_missing: bool = False) -> Dict:
        """
        取某帧结果: 已算直接返回; 非关键帧复用最近前序关键帧
        compute_if_missing=True 且超出跟踪前沿时先补算到该帧(边看边分割的情况下使用)
        """
        session = self._get_video_session(session_id)
        if compute_if_missing and frame_idx not in session.keyframe_results:
            frontier = session.last_computed_frame # 最新算完的帧
            if frontier is None or frame_idx > frontier:
                # 如果超出了最新的前序关键帧, 补算到该帧
                for _ in self.submit_video_prompts(session_id, end_frame=frame_idx):
                    pass
        res = session.keyframe_results.get(frame_idx) # 查这帧有没有算过, 算过直接返回
        if res is not None:
            return {"frame_idx": frame_idx, "keyframe": True,
                    **self._visible_result(session, frame_idx, res)}
        prev = self._latest_cached_before(session, frame_idx) # 不然的话用关键帧顶替
        if prev is None:
            raise ValueError(f"第 {frame_idx} 帧之前没有已计算的关键帧, 请先 submit")
        return {"frame_idx": frame_idx, "keyframe": False, "reused_from": prev,
                **self._visible_result(session, frame_idx, session.keyframe_results[prev])}

    def submit_video_prompts(self, session_id: str,
                             start_frame: Optional[int] = None,
                             end_frame: Optional[int] = None,
                             num_frames: Optional[int] = None) -> Generator[Dict, None, None]:
        """
        提交分割申请(生成器, 逐关键帧产出事件, 路由层逐个转发)

        范围: [start_frame, end_frame], 也可用 num_frames 只算一段
        默认 start = 跟踪前沿+1(增量续算), end = 最后一帧
        提示帧自动成为关键帧; 起算点之前的提示帧先补算(lead), 让提示进入追踪记忆
        """
        session = self._get_video_session(session_id)
        max_frame = (session.received_frame_count - 1) if session.is_streaming \
            else session.frame_count - 1
        if max_frame < 0:
            raise ValueError("还没有可计算的帧")

        # 确定计算起点 f0, 三级回退: 手动指定 > 接着上次算 > 从头算
        if start_frame is not None:
            # 1. 用户显式指定: 拖到某帧起新链 / 强制从某帧重算
            f0 = start_frame
        elif session.last_computed_frame is not None:
            # 2.  增量续算: 从跟踪前沿的下一帧开始, 不重复劳动
            #    (删除提示后前沿会回退, 这里自动从作废处重算)
            f0 = session.last_computed_frame + 1
        else:
            # 3.  全新会话首次提交: 从网格锚点开始, 无锚点则从视频开头
            f0 = session.anchor_frame if session.anchor_frame is not None else 0
        end = max_frame if end_frame is None else min(end_frame, max_frame) # 如果不指定end_frame, 则用max_frame, 分割到视频最后一帧, 由会话状态算出来的(内部)

        if num_frames is not None:
            end = min(end, f0 + num_frames - 1) # num_frames是调用方传进来的, 这次调用只想算多少帧(用户参数)
        if f0 > end:
            raise ValueError(f"计算范围为空: start={f0}, end={end}") # 开始大于结束

        prompt_frames = sorted(f for f, groups in session.frame_prompts.items()
                               if any(g.points or g.box is not None for g in groups.values())) # 将提示的帧按照帧的idx进行偏序,再送入计算

        if session.anchor_frame is None:
            session.anchor_frame = f0
        anchor = session.anchor_frame

        # TODO(v2 候选优化): 插帧网格改为"提示对齐"——每次出现新的提示帧后,
        # 网格相位重锚到最新提示帧, 而不是现在的固定锚(首次计算帧)+提示事件帧并集
        #
        # 示例(stride=5, 提示在 3、20 帧, 计算范围 0~30):
        #   现设计关键帧 = 0,3,5,10,15,20,25,30  (锚 0 的网格 ∪ 提示帧)
        #   提示对齐   = 3,8,13,18,20,25,30      (锚 3 的网格, 提示帧 20 后相位不变)
        # 计算量相近, 但后者每个计算帧离最近提示帧 <= stride, 掩码漂移更小
        # (v1 已具备"提示帧后的帧复用提示帧结果"——提示帧本身在并集里, 此处是进一步收紧)
        #
        # 注意点:
        #   1. 流式下可能反而更密: 旧相位帧已算完撤不回, 新相位继续算,
        #      两相位并集使关键帧更密(除非作废旧缓存重算, 代价更大)
        #   2. anchor 变成动态状态(每次新提示帧更新), 影响所有引用处:
        #      _invalidate_from / _latest_cached_before / 本方法的 plan 相位计算
        #      以及 reset_video_tracking 对 anchor 的清理
        #   3. 删除提示帧后相位是否回退需要定义清楚
        # 结论: v1 保留固定锚+事件帧(行为已正确), 此为 v2 候选, 改动前需过一遍上面三点

        # 计算计划 = 网格帧 ∪ 范围内的提示帧; lead = 起点之前的提示帧(先补算, 让提示进入记忆)
        plan = sorted({f for f in range(f0, end + 1) if (f - anchor) % session.frame_stride == 0}
                      | {f for f in prompt_frames if f0 <= f <= end})

        lead = [f for f in prompt_frames if f < f0] # 当显式指定的start_frame越过了某些提示帧时, 比如提示在第3帧, 但是只要10-20帧的结果， 这样的话第3帧的提示就无法输入了, 所以要加lead
        
        # 取消传播计算
        session.cancel_event.clear()
        yield {"type": "propagate_start", "start_frame": f0, "end_frame": end,
               "num_keyframes": len(plan)}
        
        # lead一定在plan前面, 因为lead < f0 <= plan
        for f in lead:
            # lead不能太稠密, 不过也不会很稠密, 极端情况暂停模式下用户在几十上百个帧各加了提示，再一次 submit, 不过这种情况下用户已经不在意
            # 实时的这个需求了, 处理一段时间也就处理一段时间了
            if session.cancel_event.is_set():
                yield {"type": "cancelled", "frame_idx": f}
                return
            self._ensure_prompt_frame_pushed(session, f)
            self._flush_dirty_frame(session, f)
            self._compute_and_cache(session, f)
        if lead:
            yield {"type": "prompts_applied", "frames": lead}
        
        for i, f in enumerate(plan):
            if session.cancel_event.is_set():
                yield {"type": "cancelled", "frame_idx": f, "progress": i / len(plan)}
                return
            self._ensure_prompt_frame_pushed(session, f)
            self._flush_dirty_frame(session, f)
            res = self._compute_and_cache(session, f)
            yield {"type": "keyframe", "frame_idx": f, "progress": (i + 1) / len(plan),
                   **self._visible_result(session, f, res)}

        yield {"type": "propagate_done", "start_frame": f0, "end_frame": end}
    
    
    # 给路由层用的公开API, 用于取消传播
    # submit 生成器跑在工作线程里(路由层 run_in_executor 放入),
    # 取消请求来自 WebSocket 所在的事件循环线程; 一个线程无法从外部
    # 安全停掉另一个线程(工作线程)里的生成器, 所以这里只置位线程安全的 cancel_event,
    # 由生成器在每帧边界自己检查并退出(协作式取消, 当前帧会算完)
    def cancel_video_propagate(self, session_id: str) -> Dict:
        """
        请求取消正在进行的 submit 传播(协作式取消)
        只是置位 cancel_event, 当前帧算完后循环自行退出并 yield cancelled
        """
        session = self._get_video_session(session_id)
        session.cancel_event.set()
        return {"success": True, "message": "取消信号已发出, 当前帧算完后停止"}
    
    def reset_video_tracking(self, session_id: str) -> Dict:
        """
        清空追踪状态(提示/物体/缓存/前沿), 但保留帧入库状态
        (流式的 stream_c2s/s2c 映射、received_frame_count、last_frame 不动,
        已入库的帧不需要重推; 底层用 clear_objects 全量清空追踪记忆)
        之后需要重新加提示并 submit
        重来一次分割, 不是重来一遍视频, session依旧存在, 这个使用场景是在流视频
        """
        session = self._get_video_session(session_id)
        session.cancel_event.set()          # 若有正在进行的 submit, 一并停掉
        self.compute_engine.clear_video_objects(session.video_session)
        session.frame_prompts.clear()
        session.dirty_prompts.clear()
        session.keyframe_results.clear()
        session.submitted_groups.clear()
        session.group_first_frame.clear()
        session.anchor_frame = None
        session.last_computed_frame = None
        return {"success": True, "message": "追踪状态已重置, 帧入库状态保留"}

    def close_video_session(self, session_id: str) -> Dict:
        """
        删除整个视频会话(先发出取消信号停掉可能的在途 submit)
        底层 inference_session 随引用释放, engine 无显式 close 原语
        """
        session = self.session_manager.get_video_session(session_id)
        if session is None:
            raise ValueError(f"会话 {session_id} 不存在")
        session.cancel_event.set() # 停掉submit
        self.session_manager.delete_video_session(session_id) # 删除视频会话
        return {"success": True, "message": f"会话 {session_id} 已删除"}
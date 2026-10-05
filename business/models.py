"""
业务层数据模型

- PointGroup: 一组点+标签(可选框)对应一个物体
- ImageSession / VideoSession: 交互式会话状态
- ImagePromptFile / VideoPromptFile: 提示文件格式(pydantic 校验)
"""

import time
import threading
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Set, Tuple, Any

import torch
from PIL import Image
from pydantic import BaseModel


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
    视频交互式会话(离线/流式共享一份状态, 字段按场景分区)

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

"""视频分割引擎: Sam3TrackerVideoModel

本文件是 SAM3 适配层: _evict_old_output 等优化与 SAM3 内部结构
(output_dict_per_obj、num_maskmem=7 等)深度耦合, 属于适配层实现细节
"""

import torch
from PIL import Image
from collections import OrderedDict
from typing import Optional, Dict, List, Tuple, Any

from transformers import Sam3TrackerVideoModel, Sam3TrackerVideoProcessor

from .base import BaseEngine


class VideoTrackerEngine(BaseEngine):
    """Sam3TrackerVideoModel 视频分割引擎"""

    def __init__(self, device: torch.device, model_path: str):
        super().__init__(device, model_path)

    def load(self):
        """加载模型"""
        if self.model is None:
            self.model = Sam3TrackerVideoModel.from_pretrained(
                self.model_path, torch_dtype=torch.bfloat16).to(self.device)
            self.processor = Sam3TrackerVideoProcessor.from_pretrained(self.model_path)

    def unload(self):
        """卸载模型"""
        if self.model is not None:
            del self.model
            del self.processor
            self.model = None
            self.processor = None

    def init_session(self, video_frames: Optional[List[Image.Image]] = None,
                        video_storage_device: Optional[str] = None,
                        frame_store: str = "ram",
                        frame_store_dir: Optional[str] = None) -> Dict:
        """
        初始化视频会话(离线/流式统一入口)

        离线：传入所有帧
        流式：传入 None

        参数:
        - video_frames: 全部帧(离线) 或 None(流式)
        - video_storage_device: 视频帧存储设备, None=跟随 inference_device(显存),
            长视频建议传 "cpu" 节省显存

        返回:
        - dict: {
            "session": inference_session,
            "video_height": int,
            "video_width": int,
            "num_frames": int
            }
        """
        if self.model is None:
            raise RuntimeError("Video 模型尚未加载")

        inference_session = self.processor.init_video_session(
            video=video_frames,
            inference_device=self.device, # 计算发生的位置(vision encoder、memory attention、mask decoder用的设备)
            inference_state_device='cpu', # 推理产物存放的位置, processed_frames 帧仓（整个视频的预处理帧）的长期存放地
            video_storage_device=video_storage_device, # processed_frames 帧仓（整个视频的预处理帧）的长期存放地
            dtype=torch.bfloat16, # 与模型权重 dtype 对齐, 帧存储/记忆特征均用 bf16(省一半显存/内存)
        ) # 这个video_frames可以是离线视频, 也可以是视频流, 

        if frame_store == "disk":
            if video_frames is not None:
                raise ValueError("disk 帧仓要求分批入库: init 时 video_frames 必须为 None,帧在 init 之后通过 add_frames 逐批追加")
            from .frame_store import DiskFrameStore
            # 偷梁换柱: 底层只用到 dict 协议(len/setitem/getitem), 无感知, 这里写的有点trick了, emmm, 不知道需不需要换掉
            inference_session.processed_frames = DiskFrameStore(
                store_dir=frame_store_dir)

        return {
            "session": inference_session,
            "video_height": inference_session.video_height,
            "video_width": inference_session.video_width,
            "num_frames": len(video_frames) if video_frames is not None else 0
        }

    def add_frame(self, session, frame: Image.Image) -> Dict:
        """
        流式场景: 只把帧注册进 session(processed_frames), 不编码不追踪
        视觉特征是惰性的: 首次对该帧 predict_frame 时才编码并按帧缓存, 且计算层推理统一用predict_frame
        (底层 add_new_frame 见 modeling_sam3_tracker_video.py:304, 仅写 processed_frames)

        返回: {"frame_idx": session帧号, "original_size": (h, w)}
        """
        if self.model is None:
            raise RuntimeError("Video 模型尚未加载")
        inference_session = session["session"]
        inputs = self.processor(images=frame, return_tensors="pt") # 变为tensor, 但不进行视觉编码
        frame_idx = inference_session.add_new_frame(
            inputs.pixel_values[0].to(torch.bfloat16)
        )
        return {"frame_idx": frame_idx,
            "original_size": tuple(inputs.original_sizes[0])}    

    def add_frames(self, session, frames: List[Image.Image]):
        """
        把一批帧注册进 session(流式传 [frame], 离线传一批), 不追踪
        统一走 video_processor 批处理路径: 与离线原生预处理一致, 批量效率高
        engine 不认识 chunk: 一次喂多少帧由业务层决定
        返回: {"frame_indices": List[int], "original_size": (h, w)}
        """
        if self.model is None:
            raise RuntimeError("Video 模型尚未加载")
        if not frames:
            raise ValueError("frames 不能为空")
        inference_session = session["session"]
        start = (len(inference_session.processed_frames)
                if inference_session.processed_frames else 0)
        processed = self.processor.video_processor(
            videos=frames, device=self.device, return_tensors="pt")
        pixel_values = processed.pixel_values_videos[0] # (T,C,H,W), 只有这批在显存
        for i in range(pixel_values.shape[0]):
            # 内部 .to(CPU, bf16), 这个来自Sam3TrackerVideoInferenceSession类的方法, 
            # 存放位置由video_storage_device这个参数决定
            inference_session.add_new_frame(pixel_values[i])

        # 竞态修复: add_new_frame 的 D2H 是 non_blocking 异步拷贝, 必须等本批
        # 拷贝全部完成, 否则下一批 video_processor 复用显存会覆写拷贝源
        torch.cuda.synchronize(pixel_values.device)


        store = inference_session.processed_frames
        if hasattr(store, "flush"):
            store.flush()  # DiskFrameStore: 每批落盘一段, 约束内存 buffer, 且仅在使用disk作为帧仓时使用
        
        h, w = processed.original_sizes[0]
        if session["video_height"] is None: # 首批记录尺寸(engine 自己建的 dict 自己维护)
            session["video_height"], session["video_width"] = h, w

        return {"frame_indices": list(range(start, start + pixel_values.shape[0])),
            "original_size": (h, w)}
    
    def add_prompt(self, session, frame_idx: int, obj_id: int,
                    click_points: Optional[List] = None,
                    click_labels: Optional[List] = None,
                    input_boxes: Optional[List] = None,
                    original_size: Optional[Tuple[int, int]] = None) -> None:
        """
        向视频指定帧添加提示(交互式入口), 也可作为细粒度提示的原语

        业务层监听用户交互后调用此方法

        输入:
        - session: 视频会话对象
        - frame_idx: 帧索引
        - obj_id: 对象ID
        - click_points: list 格式，如 [[[[x1, y1], [x2, y2]]]]
        - click_labels: list 格式，如 [[[1, 1]]]
        - input_boxes: list 格式，如 [[[x1, y1, x2, y2]]]
        - original_size: 原始图像尺寸 (height, width)
        """
        if self.model is None:
            raise RuntimeError("Video 模型尚未加载")

        inference_session = session["session"]
        
        self.processor.add_inputs_to_inference_session(
            inference_session=inference_session,
            frame_idx=frame_idx,
            obj_ids=obj_id,
            input_points=click_points,
            input_labels=click_labels,
            input_boxes=input_boxes,
            original_size=original_size
        )

    def predict_frame(self, session, frame_idx: int) -> Dict:
        """
        对 session 中已入库的指定帧做追踪推理(流式/离线统一入口)
        前提: 帧已由 add_frame 注册(流式)或 init_session 载入(离线),
        提示已由 add_prompt 登记; 调用方需保证物体数 > 0
        
        返回:
        - dict: {
            "masks": torch.Tensor,      # (num_objects, H, W)
            "shape": tuple,
            "num_objects": int,
            }
        """
        if self.model is None:
            raise RuntimeError("Video 模型尚未加载")

        inference_session = session["session"]

        outputs = self.model(
            inference_session=inference_session,
            frame_idx=frame_idx
        )

        # ---- 追踪产物瘦身(两级) ----
        # 第一级: 当步瘦身。high_res_masks 仅供本步 batched 记忆编码
        # (底层 modeling 2651 行注释), 存进输出字典后无任何读者, 直接丢弃。
        # 单帧产物 3.36MB -> 1.42MB
        for obj_outputs in inference_session.output_dict_per_obj.values():
            # 遍历每个被追踪对象的输出, output_dict_per_obj 是一个字典，键是对象 ID，值是该对象在各帧的推理结果
            # non_cond_frame_outputs：存储非条件帧（即模型自动传播预测的普通帧，而非用户点击/标注的关键帧）的输出
            # .get(frame_idx)：取出指定帧的输出数据

            entry = obj_outputs["non_cond_frame_outputs"].get(frame_idx)
            if entry is not None:
                entry.pop("high_res_masks", None)

        video_res_masks = self.processor.post_process_masks(
            [outputs.pred_masks],
            original_sizes=[[session["video_height"], session["video_width"]]],
            binarize=True
        )[0]

        # 第二级: 滑窗逐出。memory attention 只用最近6帧+条件帧,
        # object pointer 只用最近15帧(config 常量), 老输出永不再读
        self._evict_old_output(inference_session, frame_idx)

        # video_res_masks 形状: (num_objects, 1, H, W)
        video_res_masks = video_res_masks.squeeze(1)

        return {
            "masks": video_res_masks,
            "shape": video_res_masks.shape,
            "num_objects": video_res_masks.shape[0],
        }

    def _evict_old_output(self, inference_session, current_frame: int,
                               keep: int = 32) -> None:
        '''
        非条件帧输出滑窗逐出: 只保留最近 keep 帧, 老的整条删除

        依据: 追踪时只读最近15帧的object pointer + 最近6帧的记忆特征
        (config: max_object_pointers_in_encoder=16, num_maskmem=7), SAM3源码
        keep=32 留一倍余量。条件帧(cond_frame_outputs)不动

        约束:
        - 只支持前向追踪(反向追踪会回头读未来的帧, 本方法会破坏它)
        - 在老帧上补提示会触发从该帧的重追: 若其邻居帧的非条件输出已被逐出,
            缺失跳过(_gather_memory_frame_outputs:2296 对 None 跳过,
            _get_object_pointers:2367 用 .get()), 不报错;
            此时 memory attention 的输入只剩条件帧(时序证据缺失, 掩码可能轻微漂移),
            随着每帧追踪产生新输出, 记忆特征窗口6帧/指针窗口15帧内重新填满恢复
        - 归属决策(拆包已完成): 本方法与 SAM3 内部结构深度耦合, 作为适配层, 实现细节留在本文件, 不进 base.py 契约
        '''
        for obj_outputs in inference_session.output_dict_per_obj.values():
            non_cond = obj_outputs["non_cond_frame_outputs"] # 得到非条件帧
            for f in [f for f in non_cond if f < current_frame - keep]:
                del non_cond[f] # 如果非条件帧距离current_frame大于keep帧, 则被删除

    def propagate(self, session, start_frame:int = 0, end_frame: Optional[int]=None):
        """
        传播分割到帧范围(离线模式)

        这个是配合一起用的,都是先加完prompt, 然后再用这个propagate去处理, 所以这个也是很原子的操作

        参数:
        - start_frame: 起始帧索引
        - end_frame: 结束帧索引(None 表示到最后一帧)

        返回:
        - dict: {
            "frames": Dict[int, torch.Tensor],  # 每帧 (num_objects, H, W)
            "start_frame": int,
            "end_frame": int,
            "num_objects": int,
            }
        """
        if self.model is None:
            raise RuntimeError("Video 模型尚未加载")
        
        inference_session = session["session"]
        total_frames = len(inference_session.processed_frames) if inference_session.processed_frames else 0

        if end_frame is None:
            end_frame = total_frames - 1
        
        video_segments = {}
        for output in self.model.propagate_in_video_iterator(
            inference_session=inference_session,
            start_frame_idx=start_frame,
            max_frame_num_to_track=end_frame - start_frame + 1
        ):
            video_res_masks = self.processor.post_process_masks(
                [output.pred_masks],
                original_sizes=[[session["video_height"], session["video_width"]]],
                binarize=True 
            )[0]

            # 
            video_res_masks = video_res_masks.squeeze(1)
            
            video_segments[output.frame_idx] = video_res_masks

        return {
            "frames": video_segments,
            "start_frame": start_frame,
            "end_frame": end_frame,
            "num_objects": video_segments[start_frame].shape[0] if video_segments else 0,
        }

    def close_session(self, session) -> None:
        """
        释放会话资源(生命周期终点)
        这个session是SAM3在处理视频的时候自带的, 所以在处理视频的时候
        Engine也要对自己的init_session负责, 需要close掉自己的session, 这样的操作才是对偶的
        所以close_session也被视为原子操作, 但是这样做也不能保证磁盘泄露问题, 只能是在一定程度上避免

        disk 帧: 关闭句柄并删除全部段文件
        RAM 帧仓(dict): 无外部资源, 跳过, 随 inference_session 被 GC

        注意(竞态): close_video_session 先 cancel 再调本方法 在途 submit
        最多再算完当前帧; 离线传播只读不写, 已打开的 mmap 段在 unlink 后
        仍可读(Linux语义), 唯一风险是该帧恰好首读未打开的段 →NotFoundError,
        此时前端已关闭会话, 异常随 WS 断开消化, 可接受
        """
        store = session["session"].processed_frames
        if hasattr(store, "close"):
            store.close(delete=True)

    def remove_object(self, session, obj_id: int) -> bool:
        """
        删除单个被跟踪物体(底层 session 无官方 API, 此处手动删除并重排索引)
        这个已经去触碰底层的私有字典, 比较危险！！！

        已验证: 所有 per-object 状态(点/框提示、输出、maskmem 记忆特征、跟踪记录)
        均以 obj_idx 为 key 存放在 dict 中, 视觉特征缓存按帧索引与物体无关,
        因此删除并重索引是安全的; 被删物体的记忆对其他物体无影响(记忆按物体独立)

        未进行人工验证, 这对我理解SAM3的Memory机制很有用

        返回: True=删除成功, False=物体不存在
        """

        if self.model is None:
            raise RuntimeError("Video 模型尚未加载")
        
        inference_session = session["session"] # 这里需要给一下Sam3TrackerVideoProcessoer在init_video_session后返回的字段

        if obj_id not in inference_session._obj_id_to_idx:
            return False

        # 剩余 obj_id 保持相对顺序, 索引连续前移
        remaining_ids = [oid for oid in inference_session.obj_ids if oid != obj_id]
        new_id_to_idx = OrderedDict((oid, i) for i, oid in enumerate(remaining_ids))
        old_idx_to_id = inference_session._obj_idx_to_id

        def reindex(d: Dict[int, Any]) -> Dict[int, Any]:
            out = {}
            for old_i, v in d.items():
                oid = old_idx_to_id[old_i]
                if oid == obj_id:
                    continue # 如果是想要删除的object的id, 则不会保留
                out[new_id_to_idx[oid]] = v
            return out
        
        # 删除掉想要删除的object的所有信息
        inference_session.point_inputs_per_obj = reindex(inference_session.point_inputs_per_obj)
        inference_session.mask_inputs_per_obj = reindex(inference_session.mask_inputs_per_obj)
        inference_session.output_dict_per_obj = reindex(inference_session.output_dict_per_obj)
        inference_session.frames_tracked_per_obj = reindex(inference_session.frames_tracked_per_obj)
        
        # 不是很清楚这对象id和索引之间的映射关系
        # 更新对象id到索引的映射
        inference_session._obj_id_to_idx = new_id_to_idx

        # 更新索引到对象id的映射
        inference_session._obj_idx_to_id = OrderedDict((i, oid) for oid, i in new_id_to_idx.items())

        inference_session.obj_ids = remaining_ids

        # 进行防御性编程, 这个比较复杂, 我也没太看懂
        # 主要是为了防止在删除物体时, 这个物体还会有"已提交但还没来得及 forward"的提示，它的 obj_id 就会悬空留在列表里"
        # 前端之后复用同一个 group_id 新建物体时，新物体会被误判成"有新提示"，
        # 下面这个我就没看懂
        # 于是它在自己没有任何提示的帧上裸跑出垃圾 mask，而且这个 stale 条目因为没有对应输入、永远触发不了 :1814 的移除条件，会一直赖在列表里。
        if obj_id in inference_session.obj_with_new_inputs:
            inference_session.obj_with_new_inputs.remove(obj_id)
        return True

    def remove_object_inputs(self, session, obj_id: int, frame_idx: int) -> bool:
        """
        删除某物体在指定帧的点/框提示及其在该帧的输出(用于清除已提交的提示)
        
        注意: 只清这一帧; 该物体在其他帧的提示和输出不受影响

        这个方法
        """

        if self.model is None:
            raise RuntimeError("Video 模型尚未加载")

        inference_session = session["session"]
        if obj_id not in inference_session._obj_id_to_idx:
            return False
        obj_idx = inference_session._obj_id_to_idx[obj_id]

        # 去除点提示和框
        inference_session.remove_point_inputs(obj_idx, frame_idx) # 在SAM3中, 框经过processor的处理, 其实已经被处理成角点了, 这也是下一个版本需要处理的
        inference_session.remove_mask_inputs(obj_idx, frame_idx)

        # 去除掉该帧的输出
        for store in ("cond_frame_outputs", "non_cond_frame_outputs"):
            inference_session.output_dict_per_obj[obj_idx][store].pop(frame_idx, None)
        inference_session.frames_tracked_per_obj[obj_idx].pop(frame_idx, None)
        return True

    def clear_objects(self, session) -> None:
        """清空所有物体与跟踪状态(视频帧与视觉特征缓存保留)"""
        if self.model is None:
            raise RuntimeError("Video 模型尚未加载")
        session["session"].reset_tracking_data()

    
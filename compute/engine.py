"""
SAM3 纯模型计算层
与 transformers 的输入输出格式完全一致
仅进行计算, 不做过多的类型防御
"""


import torch
import numpy as np
from PIL import Image
from collections import OrderedDict
from typing import Optional, Dict, List, Tuple, Any

from transformers import (
    Sam3Model, SamProcessor,
    Sam3TrackerModel, Sam3TrackerProcessor,
    Sam3TrackerVideoModel, Sam3TrackerVideoProcessor
)


# ============ 输入校验工具 ============
# 对于Sam3TrackerProcessor和Sam3TrackerVideoProcessor的形状检测是一致的, 这个是继承的
def _validate_mask(mask: torch.Tensor) -> None:
    """校验 mask 形状"""
    if not isinstance(mask, (torch.LongTensor, torch.FloatTensor, torch.BoolTensor)):
        raise ValueError(f"mask 必须是 torch.LongTensor/torch.FloatTensor/torch.Bool, 当前 dtype: {mask.dtype}")
    if mask.ndim != 3:
        raise ValueError(f"mask 必须是 3D (batch_size, image_size, image_size)，当前: {mask.ndim}D {mask.shape}")

def _validate_points(points: torch.FloatTensor) -> None:
    """校验 points 形状"""
    if not isinstance(points, torch.FloatTensor):
        raise ValueError(f"points 必须是 torch.FloatTensor,当前类型: {type(points)}")
    if points.ndim != 4:
        raise ValueError(f"points 必须是 4D (batch_size, point_batch_size, num_points_per_image, 2)，当前: {points.ndim}D {points.shape}")
    if points.shape[-1] != 2:
        raise ValueError(f"points 最后一维必须是 2, 当前: {points.shape[-1]}")


def _validate_labels(labels: torch.LongTensor, points: torch.FloatTensor) -> None:
    """校验 labels 形状"""
    if not isinstance(labels, torch.LongTensor):
        raise ValueError(f"labels 必须是 torch.LongTensor, 当前类型: {type(labels)}")
    if labels.ndim != 3:
        raise ValueError(f"labels 必须是 3D (batch_size, point_batch_size, num_points_per_image)，当前: {labels.ndim}D {labels.shape}")
    if labels.shape != points.shape[:-1]:
        raise ValueError(f"labels 形状 {labels.shape} 与 points 形状 {points.shape[:-1]} 不匹配")


def _validate_boxes(boxes: torch.FloatTensor) -> None:
    """校验 boxes 形状"""
    if not isinstance(boxes, torch.FloatTensor):
        raise ValueError(f"boxes 必须是 torch.FloatTensor, 当前类型: {type(boxes)}")
    if boxes.ndim != 3:
        raise ValueError(f"boxes 必须是 3D (batch_size, num_boxes_per_image, 4)，当前: {boxes.ndim}D {boxes.shape}")
    if boxes.shape[-1] != 4:
        raise ValueError(f"boxes 最后一维必须是 4, 当前: {boxes.shape[-1]}")

# ============ 图像分割引擎 ===========
class ImageTrackerEngine:
    """Sam3TrackerModel 图像分割引擎"""

    def __init__(self, device: torch.device, model_path: str):
        self.device = device
        self.model_path = model_path
        self.model = None
        self.processor = None

    def load(self):
        """加载模型"""
        if self.model is None:
            self.model = Sam3TrackerModel.from_pretrained(
                self.model_path, torch_dtype=torch.bfloat16).to(self.device)
            self.processor = Sam3TrackerProcessor.from_pretrained(self.model_path)
    
    def unload(self):
        """卸载模型"""
        if self.model is not None:
            del self.model
            del self.processor
            self.model = None
            self.processor = None
    
    def predict(self, image: Image.Image,
                click_points: Optional[torch.FloatTensor] = None,
                click_labels: Optional[torch.LongTensor] = None,
                input_boxes: Optional[torch.FloatTensor] = None,
                input_masks: Optional[torch.Tensor] = None,
                image_embeddings: Optional[torch.Tensor] = None,
                original_size: Optional[Tuple[int, int]] = None) -> Dict:
        """
        图像提示分割(支持首次推理和增量推理)

        参数:
        - image: PIL.Image(首次推理时必须提供)
        - click_points: FloatTensor(1, num_objects, num_points, 2)
        - click_labels: LongTensor(1, num_objects, num_points)
        - input_boxes: FloatTensor(1, num_objects, 4)
        - input_masks: LongTensor/FloatTensor(num_objects, H, W) 已二值化
        - image_embeddings: 首次推理返回的 image_embeddings(提供则跳过 Vision Encoder)
        - original_size: 由于Image的会话是由服务层完成的, 所以必须要传original_size来解决add_prompt的问题
        返回:
        - dict: {
            "masks": torch.Tensor,      # (num_objects, H, W)
            "shape": tuple,
            "num_objects": int,
            "image_embeddings": torch.Tensor,
          }
        """
        if self.model is None:
            raise RuntimeError("Tracker 模型尚未加载")
        
        # 输入校验
        if click_points is not None:
            _validate_points(click_points)
            if click_labels is None:
                raise ValueError("提供 click_points 时必须同时提供 click_labels")
            _validate_labels(click_labels, click_points)
        if input_boxes is not None:
            _validate_boxes(input_boxes)
        if input_masks is not None:
            _validate_mask(input_masks)
        
        # 构建 processor 输入
        if image_embeddings is None:
            # 首次推理：需要 image
            if image is None:
                raise ValueError("首次推理必须提供 image")
            processor_kwargs = {"images": image, "return_tensors": 'pt'}
        else:
            # 增量推理：不需要 image
            if original_size is None:
                raise ValueError("增量推理(复用 image_embeddings)必须提供 original_size")
            processor_kwargs = {"original_sizes": [list(original_size)], "return_tensors": 'pt'} # 由于SAM3里的processor是以batch处理的,所以要再套一层列表

        if click_points is not None:
            processor_kwargs["input_points"] = click_points
        if click_labels is not None:
            processor_kwargs["input_labels"] = click_labels
        if input_boxes is not None:
            processor_kwargs["input_boxes"] = input_boxes
        
        inputs = self.processor(**processor_kwargs).to(self.device)
        # 模型权重为 bfloat16, 浮点输入需对齐 dtype; labels/original_sizes 等整型张量不动
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor) and v.is_floating_point():
                inputs[k] = v.to(torch.bfloat16)

        # 构建模型参数
        model_kwargs = {}
        if image_embeddings is not None:
            model_kwargs["image_embeddings"] = image_embeddings
        if input_masks is not None:
            model_kwargs["input_masks"] = input_masks.to(self.device, torch.bfloat16).unsqueeze(1)

        # 推理
        with torch.no_grad():
            outputs = self.model(**inputs, **model_kwargs, multimask_output=False)

        # 后处理
        masks = self.processor.post_process_masks(
            outputs.pred_masks.cpu(),
            inputs['original_sizes'],
            binarize=True
        )[0]

        # 因为multimask_output=False, 所以channel = 1, 直接squeeze, 使得masks的shape为[num_objects, height, width]
        # 单物体和多物体masks 形状都是 (num_objects, 1, H, W)
        masks = masks.squeeze(1)

        return {
            "masks": masks,
            "shape": masks.shape,
            "num_objects": masks.shape[0],
            "image_embeddings": outputs.image_embeddings,  # 返回给业务层缓存
        }
    

# ============ 视频分割引擎 ============
class VideoTrackerEngine:
    """Sam3TrackerVideoModel 视频分割引擎"""

    def __init__(self, device: torch.device, model_path: str):
        self.device = device
        self.model_path = model_path
        self.model = None
        self.processor = None

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
                     video_storage_device: Optional[str] = None) -> Dict:
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
            inference_device=self.device,
            video_storage_device=video_storage_device, # 用于存储视频帧的模型
            dtype=torch.bfloat16, # 与模型权重 dtype 对齐, 帧存储/记忆特征均用 bf16(省一半显存/内存)
        ) # 这个video_frames可以是离线视频, 也可以是视频流, 

        return {
            "session": inference_session,
            "video_height": inference_session.video_height,
            "video_width": inference_session.video_width,
            "num_frames": len(video_frames) if video_frames is not None else 0
        }

    def process_frame(self, session, frame: Image.Image) -> Dict:
        """
        流式场景：处理单帧(自动添加到 session)
        这个函数不能单独开启, 需要在有了Prompt的情况下才可以进行

        0 物体行为: transformers 在 forward 内先 add_new_frame 注册帧、再检查
        物体数, 因此 0 物体时捕获其 ValueError 并返回空结果 —— 帧正常入库
        (视觉特征在下次有物体的 forward 时才编码, 无损失), 实现"只加帧不推理"

        返回:
        - dict: {
            "mask": torch.Tensor,
            "shape": tuple,
          }
        """
        # TODO(优化): 拆分 add_frame(只编码+入session) / infer_frame(只追踪)
        # 现状: 推帧与追踪 fused, 服务层"非网格帧加提示"路径会算两遍
        #   (_push_streaming_frame 算一遍 -> flush 提示 -> predict_frame 再算一遍,
        #    第一遍结果被覆盖, 浪费一次按物体计费的追踪; 帧编码只付一次, 不重复)
        # 目标: add_frame 后 flush 提示, 再一次 infer, 省掉中间那遍追踪
        # 前提: 需确认 transformers SAM3 video 是否暴露"只加帧不推理"入口
        #  (SAM2 predictor 的 add_new_frame 只提特征, transformers 版待查) T0级的待更新!!!

        if self.model is None:
            raise RuntimeError("Video 模型尚未加载")
        
        inference_session = session["session"]

        # 处理单帧
        # 对于inputs, 包含pixel_values(torch.Tensor), original_sizes (list[list[float]]) 
        inputs = self.processor(images=frame, return_tensors="pt")

        # 流式推理
        try:
            outputs = self.model(
                inference_session=inference_session,
                frame=inputs.pixel_values[0].to(torch.bfloat16),
            )
        except ValueError as e:
            if "No objects are provided" in str(e):
                # 帧已在 add_new_frame 中注册( 发生在注册之后), 0 物体无掩码
                return {"masks": None, "shape": (0,), "num_objects": 0}
            raise

        video_res_masks = self.processor.post_process_masks(
            [outputs.pred_masks],
            original_sizes=inputs.original_sizes,
            binarize=True
        )[0]

        # video_res_masks 形状: (num_objects, 1, H, W)
        video_res_masks = video_res_masks.squeeze(1)

        return {
            "masks": video_res_masks,
            "shape": video_res_masks.shape,
            "num_objects": video_res_masks.shape[0],
        }
    
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
        分割视频指定帧(离线模式), 这个只是分割指定的帧, 并不是用来分割整个离线视频的,
        是要用propagate去分割视频段, 主要是为了分割整个视频, 用predict_frame的结果
        用于
        
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

        video_res_masks = self.processor.post_process_masks(
            [outputs.pred_masks],
            original_sizes=[[session["video_height"], session["video_width"]]],
            binarize=True
        )[0]

        # video_res_masks 形状: (num_objects, 1, H, W)
        video_res_masks = video_res_masks.squeeze(1)

        return {
            "masks": video_res_masks,
            "shape": video_res_masks.shape,
            "num_objects": video_res_masks.shape[0],
        }
    
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

# ============ 文本分割引擎 ============
# 这个暂时还比较远, 所以这个代码就只是在这占个位置, 并未经过审核
class TextPromptEngine:
    """Sam3Model 文本分割引擎"""

    def __init__(self, device: torch.device, model_path: str):
        self.device = device
        self.model_path = model_path
        self.model = None
        self.processor = None

    def load(self):
        """加载模型"""
        if self.model is None:
            self.model = Sam3Model.from_pretrained(self.model_path).to(self.device)
            self.processor = SamProcessor.from_pretrained(self.model_path)

    def unload(self):
        """卸载模型"""
        if self.model is not None:
            del self.model
            del self.processor
            self.model = None
            self.processor = None

    def predict(self, image: Image.Image, text_prompt: str,
                confidence_threshold: float = 0.5) -> Dict:
        """
        文本提示分割

        输入:
        - image: PIL.Image
        - text_prompt: str
        - confidence_threshold: float

        返回:
        - dict: {
            "masks": List[torch.Tensor],  # 每个元素 (H, W)
            "scores": List[float],
            "num_objects": int,
          }
        """
        if self.model is None:
            raise RuntimeError("Image 模型尚未加载")

        processor = SamProcessor(self.model, confidence_threshold=confidence_threshold)

        inference_state = processor.set_image(image)

        processor.reset_all_prompts(inference_state)
        inference_state = processor.set_text_prompt(prompt=text_prompt, state=inference_state)

        masks = inference_state.get('masks', [])
        scores = inference_state.get('scores', [])

        mask_list = []
        for mask in masks:
            if hasattr(mask, 'cpu'):
                mask_tensor = mask.squeeze(0).cpu()
            else:
                mask_tensor = torch.from_numpy(np.array(mask))
            mask_list.append(mask_tensor)

        return {
            "masks": mask_list,
            "scores": [float(s) for s in scores],
            "num_objects": len(mask_list),
        }
    

# ============ 主计算引擎 ============
class SAM3ComputeEngine:
    """
    SAM3 计算引擎 - 统一管理多个子引擎

    SAM3ComputeEngine
    ├── 模型加载/卸载（通用）
    ├── 图像分割(ImageTrackerEngine)
    │   └── predict()          ← 首次/增量推理统一入口
    ├── 视频分割(VideoTrackerEngine)
    │   ├── init_session()     ← 初始化会话
    │   ├── process_frame()    ← 流式处理单帧
    │   ├── add_prompt()       ← 交互式添加提示/文件添加提示
    │   ├── predict_frame()    ← 离线单帧推理
    │   └── propagate()        ← 传播推理
    └── 文本分割(TextPromptEngine)
        └── predict()
    """

    def __init__(self, 
                 model_path: str = "/root/workspace/modelRepo/SAM3",
                 device: str = "cuda:1",
                 enable_image: bool = False,
                 enable_tracker: bool = True,
                 enable_video: bool = False):
        """
        初始化 SAM3 计算引擎

        参数:
        - model_path: 模型路径
        - device: 计算设备，默认 "cuda:1"(优先 GPU)，可设为 "cpu"
        - enable_image: 是否启用文本分割模型(Sam3Model)
        - enable_tracker: 是否启用图像跟踪模型(Sam3TrackerModel)
        - enable_video: 是否启用视频跟踪模型(Sam3TrackerVideoModel)
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model_path = model_path

        # 子引擎（按需初始化并加载）
        self.image_tracker = None
        self.video_tracker = None
        self.text_prompt = None

        if enable_tracker:
            self.image_tracker = ImageTrackerEngine(self.device, model_path)
            self.image_tracker.load()
        if enable_video:
            self.video_tracker = VideoTrackerEngine(self.device, model_path)
            self.video_tracker.load()
        if enable_image:
            self.text_prompt = TextPromptEngine(self.device, model_path)
            self.text_prompt.load()
        
        # 代理映射：方法名 -> (引擎属性名, 引擎方法名)
        self._PROXY_MAP = {
            # 图像分割
            "predict_prompt": ("image_tracker", "predict"),
            # 视频分割
            "init_video_session": ("video_tracker", "init_session"),
            "process_video_frame": ("video_tracker", "process_frame"),
            "add_video_prompt": ("video_tracker", "add_prompt"),
            "predict_video_frame": ("video_tracker", "predict_frame"),
            "propagate_video": ("video_tracker", "propagate"),
            "remove_video_object": ("video_tracker", "remove_object"),
            "remove_video_object_inputs": ("video_tracker", "remove_object_inputs"),
            "clear_video_objects": ("video_tracker", "clear_objects"),
            # 文本分割
            "predict_text": ("text_prompt", "predict"),
        }

    def set_model(self, model_type: str, enabled: bool):
        """
        启用或禁用指定模型

        enabled=True: 初始化并加载到显存
        enabled=False: 从显存卸载并释放
        """
        import gc

        if model_type == "tracker":
            if enabled and self.image_tracker is None:
                self.image_tracker = ImageTrackerEngine(self.device, self.model_path)
                self.image_tracker.load()
            elif not enabled and self.image_tracker is not None:
                self.image_tracker.unload()
                del self.image_tracker
                self.image_tracker = None

        elif model_type == "video":
            if enabled and self.video_tracker is None:
                self.video_tracker = VideoTrackerEngine(self.device, self.model_path)
                self.video_tracker.load()
            elif not enabled and self.video_tracker is not None:
                self.video_tracker.unload()
                del self.video_tracker
                self.video_tracker = None

        elif model_type == "image":
            if enabled and self.text_prompt is None:
                self.text_prompt = TextPromptEngine(self.device, self.model_path)
                self.text_prompt.load()
            elif not enabled and self.text_prompt is not None:
                self.text_prompt.unload()
                del self.text_prompt
                self.text_prompt = None

        # 清理显存
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    
    def get_model_status(self) -> Dict:
        """获取当前模型状态"""
        return {
            "image": {
                "enabled": self.text_prompt is not None,
            },
            "tracker": {
                "enabled": self.image_tracker is not None,
            },
            "video": {
                "enabled": self.video_tracker is not None,
            },
            "gpu_memory_gb": torch.cuda.memory_allocated() / 1024 ** 3 if torch.cuda.is_available() else 0
        }

    def __getattr__(self, name: str):
        """动态代理到子引擎"""
        if name in self._PROXY_MAP:
            attr_name, method_name = self._PROXY_MAP[name] # 这里的attr_name指的就是子Engine
            engine = getattr(self, attr_name)
            if engine is None:
                raise RuntimeError(f"{attr_name} 模型未启用")
            return getattr(engine, method_name) # 从子引擎去获取方法
        
        # 非代理方法，抛出 AttributeError
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
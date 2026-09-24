"""图像分割引擎: Sam3TrackerModel"""

import torch
from PIL import Image
from typing import Optional, Dict, Tuple

from transformers import Sam3TrackerModel, Sam3TrackerProcessor

from .base import BaseEngine
from .validators import _validate_mask, _validate_points, _validate_labels, _validate_boxes


class ImageTrackerEngine(BaseEngine):
    """Sam3TrackerModel 图像分割引擎"""

    def __init__(self, device: torch.device, model_path: str):
        super().__init__(device, model_path)

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
    
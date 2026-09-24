"""
输入校验工具(各引擎共用)

对于Sam3TrackerProcessor和Sam3TrackerVideoProcessor的形状检测是一致的, 这个是继承的
"""


import torch


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

"""
二进制传输协议(server / client 两端共享, 格式只在这里定义一份)

WS 二进制消息(服务端 -> 客户端): 掩码包 mask bundle
  字节布局(小端):
    <B   msg_type: 1 = 掩码包(预留扩展)
    <I   frame_idx: 客户端帧号(图像场景恒为 0)
    <f   progress: 传播进度 0~1(非传播事件为 1.0)
    <I   count: 物体数
    重复 count 次:
      <I   group_id
      <I   png_len
      ...  png 字节(单通道 L, 0/255 二值)

图像载荷(双向): PNG/JPEG 字节, 直接编解码, 无额外包头
文本消息(双向): JSON, 命令与事件, 见 server.py / client.py
"""

import io
import struct
from typing import Dict, List, Optional, Tuple

from PIL import Image

# torch 仅服务端打包掩码时用到, 延迟导入(见 mask_to_png),
# 让只跑客户端的机器无需安装 torch(client.py 依赖本模块)

MSG_MASK_BUNDLE = 1

_HEADER = struct.Struct("<BIfI")
_GROUP = struct.Struct("<II")


def mask_to_png(mask: "torch.Tensor") -> bytes:
    """单物体 2D mask -> 二值 PNG 字节(>0 视为前景)"""
    import torch  # 延迟导入: 仅服务端调用
    arr = (mask.detach().cpu() > 0).to(torch.uint8).numpy() * 255
    buf = io.BytesIO()
    Image.fromarray(arr, mode="L").save(buf, format="PNG")
    return buf.getvalue()


def pack_mask_bundle(frame_idx: int, groups: List[int],
                     masks: Optional["torch.Tensor"],
                     progress: float = 1.0) -> bytes:
    """一帧的全部物体掩码打包(masks 为 None 时 count=0, groups 由 JSON 摘要携带)"""
    count = len(groups) if masks is not None else 0
    out = [_HEADER.pack(MSG_MASK_BUNDLE, frame_idx, progress, count)]
    if masks is not None:
        for row, gid in enumerate(groups):
            png = mask_to_png(masks[row])
            out.append(_GROUP.pack(gid, len(png)))
            out.append(png)
    return b"".join(out)


def unpack_mask_bundle(data: bytes) -> Tuple[int, float, Dict[int, bytes]]:
    """解包 -> (frame_idx, progress, {group_id: png_bytes})"""
    msg_type, frame_idx, progress, count = _HEADER.unpack_from(data, 0)
    if msg_type != MSG_MASK_BUNDLE:
        raise ValueError(f"未知消息类型: {msg_type}")
    offset = _HEADER.size
    masks: Dict[int, bytes] = {}
    for _ in range(count):
        gid, png_len = _GROUP.unpack_from(data, offset)
        offset += _GROUP.size
        masks[gid] = data[offset:offset + png_len]
        offset += png_len
    return frame_idx, progress, masks


def encode_image(img: Image.Image, fmt: str = "PNG") -> bytes:
    """PIL 图像 -> 字节(推帧建议 JPEG 省带宽, 图像会话用 PNG 保真)"""
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


def decode_image(data: bytes) -> Image.Image:
    """字节 -> PIL 图像(RGB)"""
    return Image.open(io.BytesIO(data)).convert("RGB")


def png_bytes_to_image(data: bytes) -> Image.Image:
    """掩码 PNG 字节 -> PIL(L 模式, 客户端显示用)"""
    return Image.open(io.BytesIO(data))

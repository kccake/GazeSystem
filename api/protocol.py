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
推帧载荷(客户端 -> 服务端): 4 字节帧号头(大端) + 图像字节,
  见 pack_push_frame / unpack_push_frame; 旧客户端发裸图像字节(无头), 兼容期内允许
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
_PUSH_HDR = struct.Struct(">I")   # 推帧帧号头(大端 4 字节)


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


def pack_push_frame(frame_idx: int, payload: bytes) -> bytes:
    """推帧消息 = 4 字节帧号头(大端) + 图像字节; 帧号由客户端编号, 服务端不再靠数消息推断"""
    return _PUSH_HDR.pack(frame_idx) + payload


def unpack_push_frame(data: bytes) -> Tuple[Optional[int], bytes]:
    """
    解推帧消息 -> (frame_idx, 图像字节); 旧客户端(裸图像字节)返回 (None, data)
    新旧协议靠魔数区分且永不冲突: 裸 JPEG=FF D8 / PNG=89 50 4E 47 开头,
    而帧号头前两字节 = idx >> 16, 需 idx >= 0xFFD80000(约 43 亿帧)才会撞 JPEG 魔数
    TODO(v2 UI 落地后): 删除旧协议兼容分支, 一律按带头解析
    """
    if data[:2] == b"\xff\xd8" or data[:4] == b"\x89PNG":
        return None, data
    if len(data) < _PUSH_HDR.size:
        raise ValueError("推帧消息过短(不足 4 字节帧号头)")
    return _PUSH_HDR.unpack_from(data, 0)[0], data[_PUSH_HDR.size:]


def decode_image(data: bytes) -> Image.Image:
    """字节 -> PIL 图像(RGB)"""
    return Image.open(io.BytesIO(data)).convert("RGB")


def png_bytes_to_image(data: bytes) -> Image.Image:
    """掩码 PNG 字节 -> PIL(L 模式, 客户端显示用)"""
    return Image.open(io.BytesIO(data))

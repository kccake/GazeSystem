"""prompt 提示文件读写(格式与 GazeSystem_v1/assets/*.json 完全一致)。"""

from __future__ import annotations

import json
from typing import Dict, List

_FORMAT_VERSION = "1.0"


def load_file(path: str) -> dict:
    """json 加载 + 基本校验(version/type/groups 或 frames 字段)。

    不合法 raise ValueError(中文消息); 合法时原样返回 dict(可直接发服务端)。
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"prompt 文件不是合法 JSON: {e}") from e
    if not isinstance(data, dict):
        raise ValueError("prompt 文件格式错误: 顶层必须是 JSON 对象")
    if "version" not in data:
        raise ValueError("prompt 文件缺少 version 字段")
    ftype = data.get("type")
    if ftype == "image":
        groups = data.get("groups")
        if not isinstance(groups, list):
            raise ValueError("prompt 文件缺少 groups 字段或不是数组")
        for g in groups:
            _check_group(g)
    elif ftype == "video":
        frames = data.get("frames")
        if not isinstance(frames, list):
            raise ValueError("prompt 文件缺少 frames 字段或不是数组")
        for fr in frames:
            if not isinstance(fr, dict) or not isinstance(fr.get("frame_idx"), int):
                raise ValueError("prompt 文件格式错误: frames 元素缺少 frame_idx")
            for g in fr.get("groups") or []:
                _check_group(g)
    else:
        raise ValueError(f"prompt 文件 type 必须是 image 或 video, 实际为: {ftype!r}")
    return data


def save_image_file(path: str, groups: List[dict]) -> None:
    """保存图像 prompt 文件。

    groups 元素: {"group_id": int, "points": [[x,y],...], "labels": [...],
                  "box": [x1,y1,x2,y2] | None}
    """
    data = {
        "version": _FORMAT_VERSION,
        "type": "image",
        "groups": [_normalize_group(g) for g in groups],
    }
    _write(path, data)


def save_video_file(path: str, frames: List[dict]) -> None:
    """保存视频 prompt 文件(空组帧跳过)。

    frames 元素: {"frame_idx": int, "groups": [...同 save_image_file...]}
    """
    out_frames = []
    for fr in frames:
        groups = [_normalize_group(g) for g in (fr.get("groups") or [])]
        if not groups:
            continue  # 空组帧跳过
        out_frames.append({"frame_idx": int(fr["frame_idx"]), "groups": groups})
    data = {"version": _FORMAT_VERSION, "type": "video", "frames": out_frames}
    _write(path, data)


def _check_group(g) -> None:
    """组基本校验: 必须是含整型 group_id 的对象(points/labels/box 允许为 null)。"""
    if not isinstance(g, dict) or not isinstance(g.get("group_id"), int):
        raise ValueError("prompt 文件格式错误: 组缺少 group_id")


def _normalize_group(g: dict) -> dict:
    """与 assets 示例对齐: 坐标转 float, 空 points/labels/box 写 null。"""
    points = g.get("points") or None
    labels = g.get("labels") or None
    box = g.get("box") or None
    return {
        "group_id": int(g["group_id"]),
        "points": [[float(p[0]), float(p[1])] for p in points] if points else None,
        "labels": [int(v) for v in labels] if labels else None,
        "box": [float(v) for v in box] if box else None,
    }


def _write(path: str, data: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

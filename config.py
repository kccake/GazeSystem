"""媒体查看器 UI 配置。"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable, List, Optional


class ActionType(Enum):
    """功能项类型。"""

    STREAM_SOURCE = "stream_source"  # 切换视频流源
    IMAGE_FOLDER = "image_folder"    # 打开图片文件夹
    VIDEO_FILE = "video_file"        # 选择本地视频文件
    ACTION = "action"                # 普通动作按钮
    SEPARATOR = "separator"          # 分隔线（无交互）


@dataclass
class FunctionItem:
    """侧边栏功能项定义。

    字段说明：
        id: 唯一标识
        item_type: 功能项类型
        icon: 显示图标（emoji 或文本）
        label: 显示文字
        payload: 携带的额外数据，例如 stream_source 的源地址
        callback: 点击后的回调函数（可选）
    """

    id: str
    item_type: ActionType
    icon: str = ""
    label: str = ""
    payload: Any = None
    callback: Optional[Callable[..., None]] = None


# 默认功能列表。个数、内容、顺序都可以动态替换。
DEFAULT_FUNCTIONS: List[FunctionItem] = [
    FunctionItem(
        id="stream_mock",
        item_type=ActionType.STREAM_SOURCE,
        icon="",
        label="模拟流",
        payload="mock",
    ),
    FunctionItem(
        id="stream_camera",
        item_type=ActionType.STREAM_SOURCE,
        icon="",
        label="摄像头",
        payload=0,
    ),
    FunctionItem(
        id="open_video_file",
        item_type=ActionType.VIDEO_FILE,
        icon="",
        label="打开视频",
        payload=None,
    ),
    FunctionItem(
        id="sep1",
        item_type=ActionType.SEPARATOR,
    ),
    FunctionItem(
        id="open_image_folder",
        item_type=ActionType.IMAGE_FOLDER,
        icon="",
        label="打开图片",
        payload=None,
    ),
    FunctionItem(
        id="sep2",
        item_type=ActionType.SEPARATOR,
    ),
    FunctionItem(
        id="snapshot",
        item_type=ActionType.ACTION,
        icon="",
        label="截图",
        payload=None,
    ),
    FunctionItem(
        id="fullscreen",
        item_type=ActionType.ACTION,
        icon="",
        label="全屏",
        payload=None,
    ),
]

# UI 样式常量
SIDEBAR_COLLAPSED_WIDTH = 48
SIDEBAR_EXPANDED_WIDTH = 180
UPDATE_INTERVAL_MS = 33  # ~30 fps

# 图片查看默认目录（运行时可改）
DEFAULT_IMAGE_FOLDER: Path = Path.home() / "Pictures"

"""图像转换与配色工具(PIL / QImage / numpy 互转, mask 着色)。"""

from __future__ import annotations

import numpy as np
from PIL import Image
from PySide6.QtGui import QColor, QImage

# 黄金角(度), 按色相轮换生成易区分颜色
_GOLDEN_ANGLE = 137.508


def pil_to_qimage(img: Image.Image) -> QImage:
    """PIL 图像 -> QImage。RGB -> Format_RGB888, L -> Format_Grayscale8。

    经 numpy 拷贝数据, 返回的 QImage 自带独立内存(不悬挂在临时数组上)。
    """
    if img.mode == "L":
        arr = np.ascontiguousarray(np.asarray(img, dtype=np.uint8))
        h, w = arr.shape
        return QImage(arr.data, w, h, w, QImage.Format_Grayscale8).copy()
    rgb = img.convert("RGB")
    arr = np.ascontiguousarray(np.asarray(rgb, dtype=np.uint8))
    h, w, _ = arr.shape
    return QImage(arr.data, w, h, w * 3, QImage.Format_RGB888).copy()


def qimage_to_pil(qimg: QImage) -> Image.Image:
    """QImage -> PIL 图像。Grayscale8 -> L, 其余按 RGB 处理。"""
    img = qimg.convertToFormat(QImage.Format_Grayscale8) \
        if qimg.format() == QImage.Format_Grayscale8 else qimg.convertToFormat(QImage.Format_RGB888)
    w, h = img.width(), img.height()
    ptr = img.bits()
    if img.format() == QImage.Format_Grayscale8:
        arr = np.frombuffer(ptr, dtype=np.uint8, count=h * img.bytesPerLine())
        arr = arr.reshape(h, img.bytesPerLine())[:, :w].copy()
        return Image.fromarray(arr, mode="L")
    arr = np.frombuffer(ptr, dtype=np.uint8, count=h * img.bytesPerLine())
    # bytesPerLine 有对齐填充, 先按行切再取有效像素
    arr = arr.reshape(h, img.bytesPerLine())[:, :w * 3].copy().reshape(h, w, 3)
    return Image.fromarray(arr, mode="RGB")


def group_color(index: int) -> QColor:
    """按黄金角色相轮换生成第 index 个组的易区分颜色。"""
    hue = ((index * _GOLDEN_ANGLE) % 360.0) / 360.0
    color = QColor()
    color.setHsvF(hue, 0.75, 0.95)
    return color


def colorize_mask(mask_qimage_gray: QImage, color: QColor, opacity: int) -> QImage:
    """灰度 mask(0/255) -> ARGB 半透明彩色叠加图(前景像素着色, 背景全透明)。"""
    gray = mask_qimage_gray.convertToFormat(QImage.Format_Grayscale8)
    w, h = gray.width(), gray.height()
    ptr = gray.bits()
    arr = np.frombuffer(ptr, dtype=np.uint8, count=h * gray.bytesPerLine())
    arr = arr.reshape(h, gray.bytesPerLine())[:, :w]
    alpha = (arr > 0).astype(np.uint8) * max(0, min(255, opacity))
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[..., 0] = color.red()
    rgba[..., 1] = color.green()
    rgba[..., 2] = color.blue()
    rgba[..., 3] = alpha
    rgba = np.ascontiguousarray(rgba)
    return QImage(rgba.data, w, h, w * 4, QImage.Format_RGBA8888).copy()

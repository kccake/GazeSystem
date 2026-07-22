"""标注画布: 原图 + mask 叠加 + 提示点/框显示, 标注交互, 缩放平移, mask 笔刷编辑。"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from PySide6.QtCore import QPointF, QRectF, QSize, Qt, Signal
from PySide6.QtGui import (QBrush, QColor, QImage, QMouseEvent, QPainter,
                           QPen, QWheelEvent)
from PySide6.QtWidgets import QWidget

from ..utils import colorize_mask, group_color

# 视图缩放范围
_MIN_SCALE = 0.1
_MAX_SCALE = 20.0
# 提示点半径(屏幕像素, 不随缩放变)
_POINT_RADIUS = 6


def _gray_to_array(qimg: QImage) -> np.ndarray:
    """灰度 QImage -> numpy(H, W) 独立拷贝。"""
    g = qimg.convertToFormat(QImage.Format_Grayscale8)
    w, h = g.width(), g.height()
    buf = np.frombuffer(g.bits(), dtype=np.uint8, count=h * g.bytesPerLine())
    return buf.reshape(h, g.bytesPerLine())[:, :w].copy()


class AnnotationCanvas(QWidget):
    """核心画布: 只负责绘制与交互, 提示点/框/mask 数据归页面管理。

    滚轮以光标为中心缩放, 中键拖拽平移;
    笔刷编辑模式(set_edit_group 非 None)下左键涂前景 255、右键擦除 0。
    """

    point_added = Signal(float, float, int)         # (x, y, label), 原图像素坐标
    box_drawn = Signal(float, float, float, float)  # (x1, y1, x2, y2), 拖拽松手发出
    mask_edited = Signal(int, object)               # (gid, 编辑后的灰度 QImage)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(320, 240)
        self.setMouseTracking(True)
        self.setCursor(Qt.CrossCursor)

        self._image: Optional[QImage] = None
        self._masks: Dict[int, QImage] = {}    # gid -> 灰度 mask(0/255)
        self._colored: Dict[int, QImage] = {}  # gid -> 着色叠加图缓存
        self._opacity = 128
        self._group_colors: Dict[int, QColor] = {}
        self._hidden_groups: Set[int] = set()
        self._points: List[Tuple[float, float, int, int]] = []          # (x, y, label, gid)
        self._boxes: List[Tuple[float, float, float, float, int]] = []  # (x1, y1, x2, y2, gid)

        self._mode = "point"              # "point" / "box"
        self._label = 1                   # button_label_mode=False 时的当前 label
        self._button_label_mode = True    # True=左键正点右键负点

        # mask 笔刷编辑
        self._edit_group: Optional[int] = None
        self._brush_radius = 20
        self._edit_arr: Optional[np.ndarray] = None
        self._painting = False
        self._paint_value = 255
        self._last_paint: Optional[QPointF] = None

        # 视图变换(图像左上角在 widget 中的偏移 + 缩放)
        self._scale = 1.0
        self._offset = QPointF(0.0, 0.0)
        self._panning = False
        self._pan_last = QPointF(0.0, 0.0)

        # box 拖拽 / 笔刷预览
        self._drag_start: Optional[QPointF] = None
        self._drag_current: Optional[QPointF] = None
        self._cursor_img: Optional[QPointF] = None

    # ---------------- 图像 / mask 数据 ----------------
    def set_image(self, img: QImage) -> None:
        """设原图。首次设置或尺寸变化时重置视图为适应窗口; 同尺寸换帧保持当前视图。"""
        need_reset = self._image is None or self._image.size() != img.size()
        self._image = img
        if need_reset:
            self.reset_view()
        else:
            self.update()

    def set_mask(self, gid: int, gray: QImage) -> None:
        """设置某组灰度 mask(0/255), 触发重着色。"""
        self._masks[gid] = gray.convertToFormat(QImage.Format_Grayscale8)
        self._colored.pop(gid, None)
        if gid == self._edit_group:
            self._edit_arr = _gray_to_array(self._masks[gid])
        self.update()

    def set_masks(self, masks: Dict[int, QImage]) -> None:
        """整体替换 mask 集({gid: 灰度 QImage})。"""
        self._masks = {gid: g.convertToFormat(QImage.Format_Grayscale8)
                       for gid, g in masks.items()}
        self._colored.clear()
        if self._edit_group is not None:
            self._edit_arr = self._load_edit_arr()
        self.update()

    def remove_mask(self, gid: int) -> None:
        """移除某组 mask。"""
        self._masks.pop(gid, None)
        self._colored.pop(gid, None)
        if gid == self._edit_group:
            self._edit_arr = self._load_edit_arr()
        self.update()

    def clear_masks(self) -> None:
        """清空全部 mask。"""
        self._masks.clear()
        self._colored.clear()
        if self._edit_group is not None:
            self._edit_arr = self._load_edit_arr()
        self.update()

    # ---------------- 显示控制 ----------------
    def set_opacity(self, v: int) -> None:
        """mask 不透明度 0-255(默认 128), 触发重着色。"""
        self._opacity = max(0, min(255, int(v)))
        self._colored.clear()
        self.update()

    def set_group_colors(self, colors: Dict[int, QColor]) -> None:
        """设置各组显示颜色(未设置的组按黄金角自动配色)。"""
        self._group_colors = dict(colors)
        self._colored.clear()
        self.update()

    def set_hidden_groups(self, gids: Set[int]) -> None:
        """隐藏的组不画 mask 和提示。"""
        self._hidden_groups = set(gids)
        self.update()

    def set_prompts(self, points: List[Tuple[float, float, int, int]],
                    boxes: List[Tuple[float, float, float, float, int]]) -> None:
        """设置提示点/框(画布只画, 数据归页面管)。

        points 元素 (x, y, label, gid); boxes 元素 (x1, y1, x2, y2, gid)。
        """
        self._points = list(points)
        self._boxes = list(boxes)
        self.update()

    # ---------------- 交互模式 ----------------
    def set_mode(self, mode: str) -> None:
        """标注模式: "point" / "box"。"""
        if mode not in ("point", "box"):
            raise ValueError(f"未知标注模式: {mode}")
        self._mode = mode
        self._drag_start = self._drag_current = None
        self.update()

    def set_label(self, label: int) -> None:
        """当前点 label: 1 正 / 0 负(button_label_mode=False 时生效)。"""
        self._label = 1 if label else 0

    def set_button_label_mode(self, on: bool) -> None:
        """True=左键正点右键负点; False=点击用当前 label。"""
        self._button_label_mode = bool(on)

    def set_edit_group(self, gid: Optional[int], brush_radius: int = 20) -> None:
        """非 None 时进入 mask 笔刷编辑模式: 左键涂前景 255、右键擦除 0,
        直接改该 gid 的灰度 mask 图, 其他标注交互禁用。"""
        if gid is not None and self._image is None:
            return  # 无原图无法编辑
        self._edit_group = gid
        self._brush_radius = max(1, int(brush_radius))
        self._painting = False
        self._last_paint = None
        self._edit_arr = self._load_edit_arr() if gid is not None else None
        self.update()

    # ---------------- 视图 ----------------
    def reset_view(self) -> None:
        """重置视图为适应窗口(保持宽高比居中)。"""
        if self._image is None:
            self._scale = 1.0
            self._offset = QPointF(0.0, 0.0)
        else:
            w, h = max(1, self.width()), max(1, self.height())
            iw, ih = self._image.width(), self._image.height()
            self._scale = min(w / iw, h / ih)
            self._offset = QPointF((w - iw * self._scale) / 2,
                                   (h - ih * self._scale) / 2)
        self.update()

    def image_size(self) -> QSize:
        """原图尺寸(未加载返回空 QSize)。"""
        return self._image.size() if self._image is not None else QSize()

    # ---------------- 坐标换算 ----------------
    def _widget_to_image(self, pos: QPointF) -> QPointF:
        return (pos - self._offset) / self._scale

    def _image_to_widget(self, pos: QPointF) -> QPointF:
        return pos * self._scale + self._offset

    def _inside_image(self, p: QPointF) -> bool:
        if self._image is None:
            return False
        return 0 <= p.x() < self._image.width() and 0 <= p.y() < self._image.height()

    def _clamp_to_image(self, p: QPointF) -> QPointF:
        if self._image is None:
            return p
        x = min(max(p.x(), 0.0), self._image.width() - 1.0)
        y = min(max(p.y(), 0.0), self._image.height() - 1.0)
        return QPointF(x, y)

    def _color_of(self, gid: int) -> QColor:
        return self._group_colors.get(gid) or group_color(gid)

    def _colored_mask(self, gid: int) -> QImage:
        """取该组的着色叠加图(带缓存, set_mask/set_opacity 时失效重建)。"""
        cached = self._colored.get(gid)
        if cached is None:
            cached = colorize_mask(self._masks[gid], self._color_of(gid), self._opacity)
            self._colored[gid] = cached
        return cached

    # ---------------- 笔刷编辑 ----------------
    def _load_edit_arr(self) -> np.ndarray:
        """加载当前编辑组的可编辑数组(无 mask 时建与原图同尺寸的全 0)。"""
        existing = self._masks.get(self._edit_group)
        if existing is not None:
            return _gray_to_array(existing)
        if self._image is not None:
            return np.zeros((self._image.height(), self._image.width()), dtype=np.uint8)
        return np.zeros((1, 1), dtype=np.uint8)

    def _apply_brush(self, img_pos: QPointF) -> None:
        """从上一笔到当前位置插值涂抹, 保证快速移动时笔迹连续。"""
        if self._edit_arr is None:
            return
        last = self._last_paint if self._last_paint is not None else img_pos
        dist = math.hypot(img_pos.x() - last.x(), img_pos.y() - last.y())
        step = max(1.0, self._brush_radius / 2)
        n = max(1, int(dist / step))
        for i in range(1, n + 1):
            t = i / n
            self._dab(last.x() + (img_pos.x() - last.x()) * t,
                      last.y() + (img_pos.y() - last.y()) * t)
        self._last_paint = QPointF(img_pos)
        self._sync_edit_mask()

    def _dab(self, x: float, y: float) -> None:
        """圆形笔刷在 (x, y) 处写入当前笔刷值。"""
        arr = self._edit_arr
        h, w = arr.shape
        r = self._brush_radius
        x0, x1 = max(0, int(x - r)), min(w, int(x + r) + 1)
        y0, y1 = max(0, int(y - r)), min(h, int(y + r) + 1)
        if x0 >= x1 or y0 >= y1:
            return
        yy, xx = np.mgrid[y0:y1, x0:x1]
        region = (xx - x) ** 2 + (yy - y) ** 2 <= r * r
        arr[y0:y1, x0:x1][region] = self._paint_value

    def _sync_edit_mask(self) -> None:
        """编辑数组 -> 该组灰度 mask 图, 失效着色缓存并重绘。"""
        gid = self._edit_group
        if gid is None or self._edit_arr is None:
            return
        h, w = self._edit_arr.shape
        self._masks[gid] = QImage(self._edit_arr.data, w, h, w,
                                  QImage.Format_Grayscale8).copy()
        self._colored.pop(gid, None)
        self.update()

    # ---------------- Qt 事件 ----------------
    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#161616"))  # 背景深色
        if self._image is None:
            painter.setPen(QColor("#888888"))
            painter.drawText(self.rect(), Qt.AlignCenter, "未加载图像")
            painter.end()
            return

        painter.setRenderHint(QPainter.SmoothPixmapTransform)
        painter.translate(self._offset)
        painter.scale(self._scale, self._scale)
        # 1) 原图
        painter.drawImage(0, 0, self._image)
        # 2) 各组半透明彩色 mask
        for gid in self._masks:
            if gid in self._hidden_groups:
                continue
            painter.drawImage(0, 0, self._colored_mask(gid))
        # 3) 提示框(组色 2px, cosmetic 笔宽不随缩放变)
        for x1, y1, x2, y2, gid in self._boxes:
            if gid in self._hidden_groups:
                continue
            pen = QPen(self._color_of(gid))
            pen.setWidth(2)
            pen.setCosmetic(True)
            painter.setPen(pen)
            painter.setBrush(Qt.NoBrush)
            painter.drawRect(QRectF(QPointF(x1, y1), QPointF(x2, y2)).normalized())
        # 4) 拖拽中的橡皮筋框
        if self._drag_start is not None and self._drag_current is not None:
            pen = QPen(QColor("#3daee9"))
            pen.setWidth(2)
            pen.setCosmetic(True)
            pen.setStyle(Qt.DashLine)
            painter.setPen(pen)
            painter.setBrush(Qt.NoBrush)
            painter.drawRect(QRectF(self._drag_start, self._drag_current).normalized())
        painter.resetTransform()

        painter.setRenderHint(QPainter.Antialiasing)
        # 5) 提示点(屏幕像素固定半径; 正=绿实心圆+白十字, 负=红实心圆+白横线)
        for x, y, label, gid in self._points:
            if gid in self._hidden_groups:
                continue
            c = self._image_to_widget(QPointF(x, y))
            color = QColor(0, 220, 0) if label == 1 else QColor(230, 40, 40)
            painter.setPen(QPen(color, 2))
            painter.setBrush(QBrush(color))
            painter.drawEllipse(c, _POINT_RADIUS, _POINT_RADIUS)
            painter.setPen(QPen(Qt.white, 2))
            painter.setBrush(Qt.NoBrush)
            if label == 1:
                painter.drawLine(c + QPointF(0, -4), c + QPointF(0, 4))
                painter.drawLine(c + QPointF(-4, 0), c + QPointF(4, 0))
            else:
                painter.drawLine(c + QPointF(-4, 0), c + QPointF(4, 0))
        # 6) 笔刷预览圈
        if self._edit_group is not None and self._cursor_img is not None:
            c = self._image_to_widget(self._cursor_img)
            pen = QPen(Qt.white)
            pen.setStyle(Qt.DashLine)
            painter.setPen(pen)
            painter.setBrush(Qt.NoBrush)
            r = self._brush_radius * self._scale
            painter.drawEllipse(c, r, r)
        painter.end()

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if self._image is None:
            return
        pos = event.position()
        btn = event.button()
        if btn == Qt.MiddleButton:  # 中键拖拽平移
            self._panning = True
            self._pan_last = pos
            self.setCursor(Qt.ClosedHandCursor)
            return
        img_pos = self._widget_to_image(pos)
        if self._edit_group is not None:  # 笔刷编辑: 左涂右擦
            if btn in (Qt.LeftButton, Qt.RightButton):
                self._painting = True
                self._paint_value = 255 if btn == Qt.LeftButton else 0
                self._last_paint = None
                self._apply_brush(img_pos)
            return
        if self._mode == "point":
            if btn == Qt.LeftButton or (self._button_label_mode and btn == Qt.RightButton):
                if not self._inside_image(img_pos):
                    return
                if self._button_label_mode:
                    label = 1 if btn == Qt.LeftButton else 0
                else:
                    label = self._label
                self.point_added.emit(img_pos.x(), img_pos.y(), label)
        elif self._mode == "box":
            if btn == Qt.LeftButton and self._inside_image(img_pos):
                self._drag_start = img_pos
                self._drag_current = img_pos
                self.update()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        pos = event.position()
        if self._panning:
            self._offset += pos - self._pan_last
            self._pan_last = pos
            self.update()
            return
        if self._image is None:
            return
        img_pos = self._widget_to_image(pos)
        if self._edit_group is not None:
            self._cursor_img = img_pos
            if self._painting:
                self._apply_brush(img_pos)
            else:
                self.update()  # 笔刷预览圈跟随
            return
        if self._drag_start is not None:
            self._drag_current = self._clamp_to_image(img_pos)
            self.update()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        btn = event.button()
        if btn == Qt.MiddleButton and self._panning:
            self._panning = False
            self.setCursor(Qt.CrossCursor)
            return
        if self._edit_group is not None:
            if btn in (Qt.LeftButton, Qt.RightButton) and self._painting:
                self._painting = False
                self._last_paint = None
                if self._edit_group in self._masks:
                    self.mask_edited.emit(self._edit_group,
                                          self._masks[self._edit_group].copy())
            return
        if self._mode == "box" and btn == Qt.LeftButton and self._drag_start is not None:
            rect = QRectF(self._drag_start, self._drag_current).normalized()
            self._drag_start = self._drag_current = None
            self.update()
            # 过小的拖拽视为误触, 不发框
            if rect.width() * self._scale >= 3 and rect.height() * self._scale >= 3:
                self.box_drawn.emit(rect.left(), rect.top(),
                                    rect.right(), rect.bottom())

    def wheelEvent(self, event: QWheelEvent) -> None:
        """滚轮以光标为中心缩放(0.1~20 倍)。"""
        if self._image is None:
            return
        factor = 1.25 if event.angleDelta().y() > 0 else 1.0 / 1.25
        new_scale = min(max(self._scale * factor, _MIN_SCALE), _MAX_SCALE)
        if new_scale == self._scale:
            return
        pos = event.position()
        ratio = new_scale / self._scale
        # 保持光标下的图像点不动
        self._offset = pos - (pos - self._offset) * ratio
        self._scale = new_scale
        self.update()

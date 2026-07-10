"""媒体查看器自定义 UI 组件。"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np
from PySide6.QtCore import QPoint, Qt, Signal
from PySide6.QtGui import (
    QBrush,
    QColor,
    QImage,
    QMouseEvent,
    QPainter,
    QPen,
    QPixmap,
)
from PySide6.QtWidgets import (
    QButtonGroup,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressDialog,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from .config import FunctionItem, ActionType, SIDEBAR_EXPANDED_WIDTH


class StreamPage(QWidget):
    """视频流显示页面。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.label = QLabel("等待视频源...")
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setStyleSheet("background-color: #1a1a1a; color: #888888; font-size: 18px;")
        self.label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.label.setScaledContents(False)
        layout.addWidget(self.label)

    def set_pixmap(self, pixmap: QPixmap):
        self.label.setPixmap(pixmap.scaled(
            self.label.size(),
            Qt.KeepAspectRatio,
            Qt.FastTransformation,
        ))

    def set_status_text(self, text: str):
        self.label.setText(text)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        current = self.label.pixmap()
        if current and not current.isNull():
            self.label.setPixmap(current.scaled(
                self.label.size(),
                Qt.KeepAspectRatio,
                Qt.FastTransformation,
            ))


class ImagePage(QWidget):
    """图片浏览页面。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._current_pixmap: QPixmap | None = None
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.label = QLabel("请选择图片文件夹")
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setStyleSheet("background-color: #1a1a1a; color: #888888; font-size: 18px;")
        self.label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self.label)

    def set_pixmap(self, pixmap: QPixmap):
        self._current_pixmap = pixmap
        self._refresh()

    def _refresh(self):
        if self._current_pixmap and not self._current_pixmap.isNull():
            self.label.setPixmap(self._current_pixmap.scaled(
                self.label.size(),
                Qt.KeepAspectRatio,
                Qt.FastTransformation,
            ))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._refresh()


class SegmentationLabel(QLabel):
    """支持点击提示的图片显示组件。"""

    point_added = Signal(float, float, int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("background-color: #1a1a1a;")
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self._original_pixmap: Optional[QPixmap] = None
        self._points: List[Tuple[float, float, int]] = []
        self._mask: Optional[np.ndarray] = None
        self._mask_alpha = 128
        self._current_label = 1
        self._point_radius = 6

    def set_image(self, pixmap: QPixmap) -> None:
        self._original_pixmap = pixmap
        self._points.clear()
        self._mask = None
        self._refresh()

    def set_current_label(self, label: int) -> None:
        self._current_label = label

    def set_mask_alpha(self, alpha: int) -> None:
        self._mask_alpha = max(0, min(255, alpha))
        self.update()

    def add_point(self, x: float, y: float, label: int) -> None:
        self._points.append((x, y, label))
        self.update()

    def clear_points(self) -> None:
        self._points.clear()
        self._mask = None
        self.update()

    def set_mask(self, mask: Optional[np.ndarray]) -> None:
        self._mask = mask
        self.update()

    def get_points_labels(self) -> Tuple[List[Tuple[float, float]], List[int]]:
        points = [(p[0], p[1]) for p in self._points]
        labels = [p[2] for p in self._points]
        return points, labels

    def _compute_geometry(self) -> Tuple[float, QPoint]:
        if self._original_pixmap is None or self._original_pixmap.isNull():
            return 1.0, QPoint(0, 0)
        label_size = self.size()
        pixmap_size = self._original_pixmap.size()
        scale = min(
            label_size.width() / max(1, pixmap_size.width()),
            label_size.height() / max(1, pixmap_size.height()),
        )
        scaled_w = pixmap_size.width() * scale
        scaled_h = pixmap_size.height() * scale
        offset_x = int((label_size.width() - scaled_w) / 2)
        offset_y = int((label_size.height() - scaled_h) / 2)
        return scale, QPoint(offset_x, offset_y)

    def _refresh(self) -> None:
        if self._original_pixmap and not self._original_pixmap.isNull():
            self.setPixmap(self._original_pixmap.scaled(
                self.size(),
                Qt.KeepAspectRatio,
                Qt.FastTransformation,
            ))

    def mousePressEvent(self, event: QMouseEvent):
        if self._original_pixmap is None or self._original_pixmap.isNull():
            super().mousePressEvent(event)
            return

        scale, offset = self._compute_geometry()
        img_x = (event.pos().x() - offset.x()) / scale
        img_y = (event.pos().y() - offset.y()) / scale
        pixmap_size = self._original_pixmap.size()

        if 0 <= img_x < pixmap_size.width() and 0 <= img_y < pixmap_size.height():
            self.add_point(img_x, img_y, self._current_label)
            self.point_added.emit(img_x, img_y, self._current_label)

    def paintEvent(self, event):
        super().paintEvent(event)
        if self._original_pixmap is None:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        scale, offset = self._compute_geometry()

        if self._mask is not None:
            mask_pixmap = self._mask_to_pixmap(self._mask, scale)
            if mask_pixmap is not None:
                painter.drawPixmap(offset, mask_pixmap)

        for x, y, label in self._points:
            px = int(x * scale + offset.x())
            py = int(y * scale + offset.y())
            color = QColor(0, 255, 0) if label == 1 else QColor(255, 0, 0)
            painter.setPen(QPen(color, 2))
            painter.setBrush(QBrush(color))
            painter.drawEllipse(
                px - self._point_radius,
                py - self._point_radius,
                self._point_radius * 2,
                self._point_radius * 2,
            )
            painter.setPen(QPen(Qt.white, 2))
            if label == 1:
                painter.drawLine(px, py - 4, px, py + 4)
                painter.drawLine(px - 4, py, px + 4, py)
            else:
                painter.drawLine(px - 4, py, px + 4, py)

    def _mask_to_pixmap(self, mask: np.ndarray, scale: float) -> Optional[QPixmap]:
        h, w = mask.shape[:2]
        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        rgba[mask > 0] = [0, 255, 0, self._mask_alpha]
        image = QImage(rgba.data, w, h, w * 4, QImage.Format_RGBA8888).copy()
        return QPixmap.fromImage(image).scaled(
            int(self._original_pixmap.width() * scale),
            int(self._original_pixmap.height() * scale),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._refresh()


class SegmentationPage(QWidget):
    """图片点击提示分割页面。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._image_path: Optional[Path] = None
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        toolbar = QWidget()
        toolbar.setFixedHeight(44)
        toolbar.setStyleSheet("""
            QWidget {
                background-color: #252526;
                border-bottom: 1px solid #3c3c3c;
            }
            QPushButton {
                background-color: #3c3c3c;
                color: #ffffff;
                border: none;
                border-radius: 4px;
                padding: 4px 12px;
                font-size: 13px;
            }
            QPushButton:hover { background-color: #505050; }
            QPushButton:pressed { background-color: #606060; }
            QRadioButton { color: #cccccc; font-size: 13px; }
            QLineEdit {
                background-color: #3c3c3c;
                color: #ffffff;
                border: 1px solid #505050;
                border-radius: 4px;
                padding: 2px 6px;
            }
            QSlider::groove:horizontal {
                height: 4px;
                background: #3c3c3c;
            }
            QSlider::handle:horizontal {
                background: #888888;
                width: 12px;
                margin: -4px 0;
            }
        """)
        toolbar_layout = QHBoxLayout(toolbar)
        toolbar_layout.setContentsMargins(8, 4, 8, 4)
        toolbar_layout.setSpacing(8)

        self.positive_btn = QRadioButton("正样本")
        self.negative_btn = QRadioButton("负样本")
        self.positive_btn.setChecked(True)
        self.mode_group = QButtonGroup(self)
        self.mode_group.addButton(self.positive_btn, 1)
        self.mode_group.addButton(self.negative_btn, 0)
        self.mode_group.idClicked.connect(self._on_mode_changed)

        self.clear_btn = QPushButton("清除点")
        self.segment_btn = QPushButton("运行分割")
        self.open_btn = QPushButton("打开图片")

        self.alpha_slider = QSlider(Qt.Horizontal)
        self.alpha_slider.setRange(0, 255)
        self.alpha_slider.setValue(128)
        self.alpha_slider.setFixedWidth(100)

        self.server_input = QLineEdit()
        self.server_input.setPlaceholderText("服务地址")
        self.server_input.setFixedWidth(160)

        toolbar_layout.addWidget(self.open_btn)
        toolbar_layout.addSpacing(12)
        toolbar_layout.addWidget(self.positive_btn)
        toolbar_layout.addWidget(self.negative_btn)
        toolbar_layout.addWidget(self.clear_btn)
        toolbar_layout.addWidget(self.segment_btn)
        toolbar_layout.addSpacing(12)
        toolbar_layout.addWidget(QLabel("透明度:"))
        toolbar_layout.addWidget(self.alpha_slider)
        toolbar_layout.addStretch()
        toolbar_layout.addWidget(QLabel("服务:"))
        toolbar_layout.addWidget(self.server_input)

        layout.addWidget(toolbar)

        self.label = SegmentationLabel()
        layout.addWidget(self.label, stretch=1)

        self.status_label = QLabel("请选择图片并点击目标区域")
        self.status_label.setStyleSheet(
            "background-color: #1a1a1a; color: #888888; padding: 4px 8px;"
        )
        self.status_label.setFixedHeight(26)
        layout.addWidget(self.status_label)

        self.open_btn.clicked.connect(self._choose_image)
        self.clear_btn.clicked.connect(self._clear_points)
        self.segment_btn.clicked.connect(self._request_segment)
        self.alpha_slider.valueChanged.connect(self.label.set_mask_alpha)

    def set_server_url(self, url: str) -> None:
        self.server_input.setText(url)

    def get_server_url(self) -> str:
        return self.server_input.text().strip()

    def load_image(self, path: Path) -> bool:
        pixmap = QPixmap(str(path))
        if pixmap.isNull():
            QMessageBox.warning(self, "提示", f"无法加载图片: {path}")
            return False
        self._image_path = path
        self.label.set_image(pixmap)
        self.status_label.setText(f"已加载: {path.name}")
        return True

    def get_image_path(self) -> Optional[Path]:
        return self._image_path

    def get_points_labels(self) -> Tuple[List[Tuple[float, float]], List[int]]:
        return self.label.get_points_labels()

    def set_mask(self, mask: Optional[np.ndarray]) -> None:
        self.label.set_mask(mask)

    def set_status(self, text: str) -> None:
        self.status_label.setText(text)

    def _on_mode_changed(self, mode_id: int) -> None:
        self.label.set_current_label(mode_id)

    def _choose_image(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择图片",
            str(Path.home()),
            "图片文件 (*.jpg *.jpeg *.png *.bmp *.webp *.tiff);;所有文件 (*.*)",
        )
        if file_path:
            self.load_image(Path(file_path))

    def _clear_points(self) -> None:
        self.label.clear_points()
        self.status_label.setText("已清除点击点")

    def _request_segment(self) -> None:
        self.segment_btn.setEnabled(False)
        self.status_label.setText("正在请求后端分割...")


class StackedDisplay(QStackedWidget):
    """主显示区：在视频流和图片之间切换。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.stream_page = StreamPage(self)
        self.image_page = ImagePage(self)
        self.segmentation_page = None # 新加的功能
        self.addWidget(self.stream_page)
        self.addWidget(self.image_page)

    def set_segmentation_page(self, page):
        """由 MainWindow 注入交互分割页面，避免循环导入。"""
        if self.segmentation_page is not None:
            self.removeWidget(self.segmentation_page)
        self.segmentation_page = page
        self.addWidget(self.segmentation_page)

    def show_stream(self):
        self.setCurrentWidget(self.stream_page)

    def show_image(self):
        self.setCurrentWidget(self.image_page)

    def show_segmentation(self):
        if self.segmentation_page is not None:
            self.setCurrentWidget(self.segmentation_page)

    def set_stream_pixmap(self, pixmap: QPixmap):
        self.stream_page.set_pixmap(pixmap)

    def set_stream_status(self, text: str):
        self.stream_page.set_status_text(text)

    def set_image_pixmap(self, pixmap: QPixmap):
        self.image_page.set_pixmap(pixmap)


class SidebarWidget(QWidget):
    """抽屉式侧边栏：默认隐藏，从侧面滑出。"""

    item_clicked = Signal(FunctionItem)
    close_clicked = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._functions: List[FunctionItem] = []
        self._buttons: List[QPushButton] = []
        self._setup_ui()
        self.setFixedWidth(SIDEBAR_EXPANDED_WIDTH)

    def _setup_ui(self):
        self.setStyleSheet("""
            SidebarWidget {
                background-color: #252526;
                border-right: 1px solid #3c3c3c;
            }
            #sidebar_header {
                background-color: #333333;
                border-bottom: 1px solid #3c3c3c;
            }
            #sidebar_title {
                color: #ffffff;
                font-size: 14px;
                font-weight: bold;
            }
            #close_btn {
                color: #cccccc;
                font-size: 16px;
                font-weight: bold;
            }
            #close_btn:hover {
                color: #ffffff;
                background-color: #e81123;
            }
            QPushButton {
                border: none;
                background-color: transparent;
                color: #cccccc;
                text-align: left;
                padding: 10px 12px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #3c3c3c;
            }
            QPushButton:pressed {
                background-color: #505050;
            }
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # 顶部标题栏
        header = QWidget()
        header.setObjectName("sidebar_header")
        header.setFixedHeight(36)
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(12, 0, 4, 0)
        header_layout.setSpacing(0)

        title = QLabel("功能")
        title.setObjectName("sidebar_title")

        self.close_btn = QPushButton("×")
        self.close_btn.setObjectName("close_btn")
        self.close_btn.setFixedSize(28, 28)
        self.close_btn.clicked.connect(self.close_clicked.emit)

        header_layout.addWidget(title)
        header_layout.addStretch()
        header_layout.addWidget(self.close_btn)
        layout.addWidget(header)

        # 可滚动的功能列表
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        self.content = QWidget()
        self.content_layout = QVBoxLayout(self.content)
        self.content_layout.setContentsMargins(4, 8, 4, 8)
        self.content_layout.setSpacing(4)
        self.content_layout.addStretch()

        scroll.setWidget(self.content)
        layout.addWidget(scroll)

    def set_functions(self, functions: List[FunctionItem]):
        """根据配置动态重建功能按钮。"""
        self._functions = list(functions)
        # 清除旧按钮（保留最后的 stretch）
        while self.content_layout.count() > 1:
            item = self.content_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()
        self._buttons.clear()

        for func in self._functions:
            if func.item_type == ActionType.SEPARATOR:
                line = QFrame()
                line.setFrameShape(QFrame.HLine)
                line.setStyleSheet("color: #3c3c3c;")
                line.setFixedHeight(2)
                self.content_layout.insertWidget(self.content_layout.count() - 1, line)
                continue

            btn = QPushButton(func.label)
            btn.setToolTip(func.label)
            btn.setFixedHeight(40)
            btn.clicked.connect(lambda checked=False, f=func: self.item_clicked.emit(f))
            self.content_layout.insertWidget(self.content_layout.count() - 1, btn)
            self._buttons.append(btn)


class StatusBar(QWidget):
    """顶部状态栏。"""

    menu_clicked = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(36)
        self.setStyleSheet("""
            StatusBar {
                background-color: #333333;
                color: #ffffff;
                border-bottom: 1px solid #3c3c3c;
            }
            QLabel {
                color: #ffffff;
                padding: 0 10px;
            }
            QPushButton#menu_btn {
                background-color: #444444;
                color: #ffffff;
                border: none;
                border-radius: 4px;
                padding: 4px 12px;
                font-size: 13px;
            }
            QPushButton#menu_btn:hover {
                background-color: #555555;
            }
        """)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 0, 8, 0)
        layout.setSpacing(12)

        self.menu_btn = QPushButton("☰ 功能")
        self.menu_btn.setObjectName("menu_btn")
        self.menu_btn.setFixedHeight(26)
        self.menu_btn.clicked.connect(self.menu_clicked.emit)

        self.mode_label = QLabel("模式: 视频流")
        self.status_label = QLabel("状态: 待机")
        self.info_label = QLabel("")

        layout.addWidget(self.menu_btn)
        layout.addWidget(self.mode_label)
        layout.addWidget(self.status_label)
        layout.addStretch()
        layout.addWidget(self.info_label)

    def set_mode(self, mode: str):
        self.mode_label.setText(f"模式: {mode}")

    def set_status(self, status: str):
        self.status_label.setText(f"状态: {status}")

    def set_info(self, info: str):
        self.info_label.setText(info)


class ControlBar(QWidget):
    """底部控制条。"""

    play_pause_clicked = Signal()
    prev_clicked = Signal()
    next_clicked = Signal()
    snapshot_clicked = Signal()
    fullscreen_clicked = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(48)
        self.setStyleSheet("""
            ControlBar {
                background-color: #252526;
                border-top: 1px solid #3c3c3c;
            }
            QPushButton {
                background-color: #3c3c3c;
                color: #ffffff;
                border: none;
                border-radius: 4px;
                padding: 6px 14px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #505050;
            }
        """)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 0, 12, 0)
        layout.setSpacing(8)

        self.play_btn = QPushButton("⏸ 暂停")
        self.prev_btn = QPushButton("⏮ 上一张")
        self.next_btn = QPushButton("下一张 ⏭")
        self.snapshot_btn = QPushButton("📸 截图")
        self.fullscreen_btn = QPushButton("⛶ 全屏")

        self.play_btn.clicked.connect(self.play_pause_clicked.emit)
        self.prev_btn.clicked.connect(self.prev_clicked.emit)
        self.next_btn.clicked.connect(self.next_clicked.emit)
        self.snapshot_btn.clicked.connect(self.snapshot_clicked.emit)
        self.fullscreen_btn.clicked.connect(self.fullscreen_clicked.emit)

        layout.addWidget(self.play_btn)
        layout.addWidget(self.prev_btn)
        layout.addWidget(self.next_btn)
        layout.addStretch()
        layout.addWidget(self.snapshot_btn)
        layout.addWidget(self.fullscreen_btn)

        self.set_mode("stream")

    def set_mode(self, mode: str):
        """根据当前模式显示/隐藏对应按钮。"""
        is_stream = mode == "stream"
        self.play_btn.setVisible(is_stream)
        self.prev_btn.setVisible(not is_stream)
        self.next_btn.setVisible(not is_stream)

    def set_play_text(self, text: str):
        self.play_btn.setText(text)

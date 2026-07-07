"""媒体查看器自定义 UI 组件。"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, List

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
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


class StackedDisplay(QStackedWidget):
    """主显示区：在视频流和图片之间切换。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.stream_page = StreamPage(self)
        self.image_page = ImagePage(self)
        self.addWidget(self.stream_page)
        self.addWidget(self.image_page)

    def show_stream(self):
        self.setCurrentWidget(self.stream_page)

    def show_image(self):
        self.setCurrentWidget(self.image_page)

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

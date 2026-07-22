"""主窗口外壳: 左侧导航 + 堆叠页面 + 状态栏。"""

from __future__ import annotations

from PySide6.QtWidgets import (QHBoxLayout, QLabel, QListWidget, QMainWindow,
                               QStackedWidget, QWidget)

from .context import AppContext
from .pages.home_page import HomePage
from .pages.image_page import ImagePage
from .pages.video_offline_page import VideoOfflinePage
from .pages.video_stream_page import VideoStreamPage

# 导航项(顺序即页面顺序, 首页恒为 0)
_NAV_TITLES = ["首页", "图像分割", "视频离线标注", "视频实时流"]


class MainWindow(QMainWindow):
    """应用外壳: 导航切换页面, 未连接时禁止进入业务页。"""

    def __init__(self, context: AppContext, parent=None):
        super().__init__(parent)
        self._ctx = context
        self.setWindowTitle("GazeSystem v1 — SAM3 交互分割")
        self.setMinimumSize(1100, 700)
        self.resize(1280, 800)
        self._setup_ui()
        context.connection_changed.connect(self._on_connection_changed)
        context.status_message.connect(
            lambda msg: self.statusBar().showMessage(msg, 5000))
        self._on_connection_changed(context.connected)

    def _setup_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._nav = QListWidget()
        self._nav.setFixedWidth(170)
        self._nav.addItems(_NAV_TITLES)
        self._nav.setStyleSheet("""
            QListWidget {
                background-color: #252526;
                border-right: 1px solid #3c3c3c;
                color: #cccccc;
                font-size: 14px;
                outline: none;
            }
            QListWidget::item { padding: 12px 14px; }
            QListWidget::item:selected { background-color: #3a3d41; color: #ffffff; }
            QListWidget::item:hover { background-color: #333333; }
        """)

        self._stack = QStackedWidget()
        self._pages = [
            HomePage(self._ctx),
            ImagePage(self._ctx),
            VideoOfflinePage(self._ctx),
            VideoStreamPage(self._ctx),
        ]
        for page in self._pages:
            self._stack.addWidget(page)

        layout.addWidget(self._nav)
        layout.addWidget(self._stack, stretch=1)

        self._conn_label = QLabel("未连接")
        self.statusBar().addPermanentWidget(self._conn_label)
        self.statusBar().showMessage("就绪")

        self._nav.currentRowChanged.connect(self._on_nav_changed)
        self._nav.setCurrentRow(0)

    def _on_nav_changed(self, row: int) -> None:
        if row > 0 and not self._ctx.connected:
            # 未连接禁止进入业务页: 阻止切换并提示
            self._nav.blockSignals(True)
            self._nav.setCurrentRow(self._stack.currentIndex())
            self._nav.blockSignals(False)
            self.statusBar().showMessage("请先连接服务器", 3000)
            return
        if 0 <= row < self._stack.count():
            self._stack.setCurrentIndex(row)

    def _on_connection_changed(self, ok: bool) -> None:
        self._conn_label.setText(f"已连接 {self._ctx.server_url}" if ok else "未连接")
        if not ok and self._stack.currentIndex() != 0:
            self._nav.setCurrentRow(0)  # 连接断开: 退回首页

    def closeEvent(self, event) -> None:
        for page in self._pages:
            shutdown = getattr(page, "shutdown", None)
            if callable(shutdown):
                try:
                    shutdown()
                except Exception:
                    pass
        super().closeEvent(event)

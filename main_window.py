"""媒体查看器主窗口。"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import List

from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QKeyEvent, QPixmap
from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QMainWindow,
    QMessageBox,
    QVBoxLayout,
    QWidget,
)

from .config import DEFAULT_FUNCTIONS, DEFAULT_IMAGE_FOLDER, FunctionItem, ActionType
from .stream_worker import StreamWorker
from .widgets import ControlBar, SidebarWidget, StackedDisplay, StatusBar


class MainWindow(QMainWindow):
    """主窗口：整合显示区、侧边栏、状态栏、控制条。"""

    def __init__(self, functions: List[FunctionItem] | None = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("GazeSystem")
        self.setMinimumSize(900, 600)
        self.resize(1280, 720)

        self._functions = functions if functions is not None else DEFAULT_FUNCTIONS
        self._current_mode = "stream"  # "stream" or "image"
        self._stream_worker: StreamWorker | None = None

        self._image_folder: Path = DEFAULT_IMAGE_FOLDER
        self._image_paths: List[Path] = []
        self._image_index = -1

        self._setup_ui()
        self._connect_signals()
        self._start_stream("mock")

    def _setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        self.setStyleSheet("background-color: #1e1e1e;")

        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # 侧边栏（抽屉式，默认隐藏）
        self.sidebar = SidebarWidget()
        self.sidebar.set_functions(self._functions)
        self.sidebar.hide()
        main_layout.addWidget(self.sidebar)

        # 右侧区域
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(0)

        self.status_bar = StatusBar()
        self.display = StackedDisplay()
        self.control_bar = ControlBar()

        right_layout.addWidget(self.status_bar)
        right_layout.addWidget(self.display, stretch=1)
        right_layout.addWidget(self.control_bar)

        main_layout.addWidget(right_widget, stretch=1)

    def _connect_signals(self):
        self.sidebar.item_clicked.connect(self._on_function_clicked)
        self.sidebar.close_clicked.connect(self.sidebar.hide)
        self.status_bar.menu_clicked.connect(self._toggle_sidebar)

        self.control_bar.play_pause_clicked.connect(self._toggle_play_pause)
        self.control_bar.prev_clicked.connect(self._show_prev_image)
        self.control_bar.next_clicked.connect(self._show_next_image)
        self.control_bar.snapshot_clicked.connect(self._take_snapshot)
        self.control_bar.fullscreen_clicked.connect(self._toggle_fullscreen)

    # ------------------------------------------------------------------
    # 视频流相关
    # ------------------------------------------------------------------
    def _start_stream(self, source):
        print(f"[MainWindow] 开始启动视频源: {source}")
        self.status_bar.set_info(f"源: {source}")
        self._stop_stream()
        self._switch_mode("stream")

        self._stream_worker = StreamWorker(source=source, parent=self)
        self._stream_worker.frame_captured.connect(self.display.set_stream_pixmap)
        self._stream_worker.status_changed.connect(self._on_stream_status_changed)
        self._stream_worker.fps_updated.connect(self._on_fps_updated)
        self._stream_worker.start()

    def _on_stream_status_changed(self, status: str):
        print(f"[StreamWorker] 状态: {status}")
        self.status_bar.set_status(status)
        if status.startswith("错误"):
            self.display.set_stream_status(status)

    def _on_fps_updated(self, fps: float):
        self.status_bar.set_info(f"FPS: {fps:.1f}")

    def _stop_stream(self):
        if self._stream_worker is not None:
            self._stream_worker.stop()
            self._stream_worker = None

    def _toggle_play_pause(self):
        if self._stream_worker is None:
            return
        if self._stream_worker.is_paused():
            self._stream_worker.resume()
            self.control_bar.set_play_text("⏸ 暂停")
        else:
            self._stream_worker.pause()
            self.control_bar.set_play_text("▶ 播放")

    # ------------------------------------------------------------------
    # 图片浏览相关
    # ------------------------------------------------------------------
    def _open_video_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择视频文件",
            str(Path.home()),
            "视频文件 (*.mp4 *.avi *.mkv *.mov *.wmv *.flv *.webm);;所有文件 (*.*)",
        )
        print(f"[MainWindow] 选择视频文件: {file_path}")
        if not file_path:
            return
        self._start_stream(file_path)

    def _open_image_folder(self):
        folder = QFileDialog.getExistingDirectory(
            self,
            "选择图片文件夹",
            str(self._image_folder),
        )
        if not folder:
            return

        self._image_folder = Path(folder)
        self._load_images_from_folder(self._image_folder)

    def _load_images_from_folder(self, folder: Path):
        extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp", ".tiff"}
        self._image_paths = sorted(
            [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in extensions]
        )
        self._image_index = 0 if self._image_paths else -1

        if not self._image_paths:
            QMessageBox.information(self, "提示", "该文件夹下没有找到支持的图片文件。")
            return

        self._switch_mode("image")
        self._show_image_at_index(0)

    def _show_image_at_index(self, index: int):
        if not self._image_paths or not (0 <= index < len(self._image_paths)):
            return

        self._image_index = index
        path = self._image_paths[index]
        pixmap = QPixmap(str(path))
        self.display.set_image_pixmap(pixmap)
        self.status_bar.set_info(f"{index + 1} / {len(self._image_paths)}  {path.name}")

    def _show_prev_image(self):
        if self._image_paths:
            new_index = (self._image_index - 1) % len(self._image_paths)
            self._show_image_at_index(new_index)

    def _show_next_image(self):
        if self._image_paths:
            new_index = (self._image_index + 1) % len(self._image_paths)
            self._show_image_at_index(new_index)

    # ------------------------------------------------------------------
    # 通用功能
    # ------------------------------------------------------------------
    def _switch_mode(self, mode: str):
        self._current_mode = mode
        self.control_bar.set_mode(mode)
        if mode == "stream":
            self.status_bar.set_mode("视频流")
            self.display.show_stream()
        else:
            self.status_bar.set_mode("图片浏览")
            self.display.show_image()

    def _on_function_clicked(self, item: FunctionItem):
        if item.item_type == ActionType.STREAM_SOURCE:
            self._start_stream(item.payload)
        elif item.item_type == ActionType.IMAGE_FOLDER:
            self._open_image_folder()
        elif item.item_type == ActionType.VIDEO_FILE:
            self._open_video_file()
        elif item.item_type == ActionType.ACTION:
            if item.id == "snapshot":
                self._take_snapshot()
            elif item.id == "fullscreen":
                self._toggle_fullscreen()
            elif item.callback is not None:
                item.callback()

    def _toggle_sidebar(self):
        if self.sidebar.isVisible():
            self.sidebar.hide()
        else:
            self.sidebar.show()

    def _take_snapshot(self):
        pixmap: QPixmap | None = None
        if self._current_mode == "stream":
            pixmap = self.display.stream_page.label.pixmap()
        else:
            pixmap = self.display.image_page.label.pixmap()

        if pixmap is None or pixmap.isNull():
            QMessageBox.information(self, "截图", "当前没有可保存的画面。")
            return

        save_dir = Path.home() / "Pictures" / "MediaViewerSnapshots"
        save_dir.mkdir(parents=True, exist_ok=True)
        filename = datetime.now().strftime("snapshot_%Y%m%d_%H%M%S_%f") + ".png"
        save_path = save_dir / filename

        pixmap.save(str(save_path))
        self.status_bar.set_info(f"截图已保存: {save_path}")

    def _toggle_fullscreen(self):
        if self.isFullScreen():
            self.showNormal()
            self.status_bar.show()
            self.control_bar.show()
            self.sidebar.show()
        else:
            self.showFullScreen()
            self.status_bar.hide()
            self.control_bar.hide()
            self.sidebar.hide()

    def keyPressEvent(self, event: QKeyEvent):
        if event.key() == Qt.Key_Escape and self.isFullScreen():
            self._toggle_fullscreen()
        elif event.key() in (Qt.Key_Left, Qt.Key_PageUp):
            self._show_prev_image()
        elif event.key() in (Qt.Key_Right, Qt.Key_PageDown, Qt.Key_Space):
            if self._current_mode == "image":
                self._show_next_image()
            else:
                self._toggle_play_pause()
        else:
            super().keyPressEvent(event)

    def closeEvent(self, event):
        self._stop_stream()
        event.accept()

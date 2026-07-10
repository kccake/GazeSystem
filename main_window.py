"""媒体查看器主窗口。"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import List

from PySide6.QtCore import Qt, QThread
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
from .interactive_segment_client import InteractiveSegmentationPage, SegmentationWorker

# 先这么用吧, 后面再说UI弹提示框的事
# 项目目录下的 masks/ 文件夹
MASK_DIR = Path(__file__).parent / "masks"  # GazeSystem/masks/
MASK_DIR.mkdir(parents=True, exist_ok=True)


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
        self._setup_interactive_segmentation() # 新增
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

        # 如果在交互分割模式，同步切换当前图片
        if self._current_mode == "segmentation":
            self._segment_worker.close_session()
            self.segmentation_page.set_paused_state(False) # 将状态set为暂停
            self.segmentation_page.load_image(path)
        
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
    # 交互式点击分割
    # ------------------------------------------------------------------
    # 这个是用来将UI与工作线程之间的信号和槽函数, 不涉及到点击的逻辑
    def _setup_interactive_segmentation(self):
        self.segmentation_page = InteractiveSegmentationPage(self)
        self.display.set_segmentation_page(self.segmentation_page)

        self._segment_worker = SegmentationWorker("http://127.0.0.1:8000")
        self._segment_thread = QThread(self)
        self._segment_worker.moveToThread(self._segment_thread)
        self._segment_thread.start()

        # 工作线程 -> UI
        self._segment_worker.started.connect(self._on_segment_session_started)
        self._segment_worker.point_result.connect(self._on_segment_point_result) # 连接到UI的槽函数上,将结果返回到函数上
        self._segment_worker.paused.connect(self._on_segment_paused)
        self._segment_worker.resumed.connect(self._on_segment_resumed)
        self._segment_worker.cleared.connect(self._on_segment_cleared)
        self._segment_worker.saved.connect(self._on_segment_saved)
        self._segment_worker.error.connect(self._on_segment_error)
        self._segment_worker.status.connect(self.segmentation_page.set_status)

        # UI -> 工作线程
        self.segmentation_page.image_loaded.connect(self._on_segment_image_loaded)
        self.segmentation_page.point_added.connect(self._on_segment_point_added) # 将交互传入的点给到工作线程的函数
        self.segmentation_page.clear_requested.connect(self._on_segment_clear_requested)
        self.segmentation_page.pause_requested.connect(self._on_segment_pause_requested)
        self.segmentation_page.resume_requested.connect(self._on_segment_resume_requested)
        self.segmentation_page.save_requested.connect(self._on_segment_save_requested)
        self.segmentation_page.back_requested.connect(self._exit_interactive_segmentation)

    # 后面又定义了很多函数来去做槽和信号的包装
    # 用于点击的交互, 这个交互套的是worker和ui, 为后续的点击逻辑做交互
    def _enter_interactive_segmentation(self):
        if not self._image_paths:
            QMessageBox.information(self, "交互分割", "请先打开图片文件夹。")
            return

        self._stop_stream()
        self._current_mode = "segmentation"
        self.status_bar.set_mode("交互分割")
        self.control_bar.set_mode("image")   # 复用上一张/下一张按钮
        self.display.show_segmentation()
        self._show_image_at_index(self._image_index)
    
    def _exit_interactive_segmentation(self):
        self._segment_worker.close_session()
        self.segmentation_page.set_paused_state(False)
        self._switch_mode("image")
        self._show_image_at_index(self._image_index)
    
    def _on_segment_image_loaded(self, path: Path):
        self._segment_worker.close_session()
        self._segment_worker.set_base_url(self.segmentation_page.get_server_url())
        self._segment_worker.start_session(str(path))

    def _on_segment_session_started(self, session_id: str):
        self.segmentation_page.set_status(f"会话已启动: {session_id[:8]}...")
    
    def _on_segment_point_added(self, x: float, y: float, label: int):
        self._segment_worker.add_point(x, y, label)

    def _on_segment_point_result(self, mask, num_points: int, message: str):
        self.segmentation_page.set_mask(mask)
        self.segmentation_page.set_status(f"{message} | 点数: {num_points}")

    def _on_segment_clear_requested(self):
        self._segment_worker.clear()
    
    # 这个就有点意义不明了
    def _on_segment_cleared(self, message: str):
        self.segmentation_page.clear_local()
        self.segmentation_page.set_paused_state(False) # 将当前状态设置为暂停
        self.segmentation_page.set_status(message) # 启动新会话

    # 一个动作会有两个函数, 用于区分动作, _on_segment_pause_requested由UI层发起, 告诉工作线程用户想暂停
    # 而_on_segment_paused则是由工作线程发起, 接收信号, 通知UI已经暂停了
    def _on_segment_pause_requested(self):
        self._segment_worker.pause()

    def _on_segment_paused(self, message: str):
        self.segmentation_page.set_paused_state(True)
        self.segmentation_page.set_status(message)

    def _on_segment_resume_requested(self):
        self._segment_worker.resume()
    
    def _on_segment_resumed(self, message: str):
        self.segmentation_page.set_paused_state(False)
        self.segmentation_page.set_status(message)
    
    def _on_segment_save_requested(self):
        self._segment_worker.save("mask")

    def _on_segment_saved(self, mask, shape, mask_base64, message):
        if mask is None:
            self.segmentation_page.set_status("没有可保存的分割结果")
            return

        default_name = f"mask_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.png"
        save_path, _ = QFileDialog.getSaveFileName(
            self,
            "保存分割掩码",
            str(MASK_DIR / default_name),
            "PNG 图片 (*.png);;所有文件 (*.*)",
        )
        if not save_path:
            return

        h, w = mask.shape
        image = QImage(mask.data, w, h, w, QImage.Format_Grayscale8).copy()
        if image.save(save_path):
            self.segmentation_page.set_status(f"掩码已保存: {save_path}")
        else:
            self.segmentation_page.set_status("保存失败")

    def _on_segment_error(self, message: str):
        self.segmentation_page.set_status(f"错误: {message}")

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
        # 这个具体的点击的一系列的触发应该要到widget去看, 主要靠widget.py下的SegmentationLabel.mousePressEvent()接收点击操作
        if item.item_type == ActionType.STREAM_SOURCE:
            self._start_stream(item.payload)
        elif item.item_type == ActionType.IMAGE_FOLDER:
            self._open_image_folder()
        elif item.item_type == ActionType.VIDEO_FILE:
            self._open_video_file()
        elif item.item_type == ActionType.ACTION:
            # 新加的交互式分割
            if item.id == "interactive_segment":
                self._enter_interactive_segmentation()
            elif item.id == "snapshot":
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
        elif self._current_mode == "segmentation":
            pixmap = self.segmentation_page.label.grab()
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

        if getattr(self, "_segment_worker", None) is not None:
            self._segment_worker.close_session()

        if getattr(self, "_segment_thread", None) is not None:
            self._segment_thread.quit() # 通知工作线程退出事件循环
            self._segment_thread.wait(2000) # 多等两秒让线程退出

        event.accept()

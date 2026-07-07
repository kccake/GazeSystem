"""独立的视频流捕获线程，不依赖 video_agent。"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Union

import cv2
import numpy as np
from PySide6.QtCore import QThread, Signal
from PySide6.QtGui import QImage, QPixmap


class StreamWorker(QThread):
    """在后台线程中捕获视频流并发出 QPixmap。"""

    frame_captured = Signal(QPixmap)
    status_changed = Signal(str)
    fps_updated = Signal(float)  # 实际渲染帧率

    def __init__(
        self,
        source: Union[int, str] = "mock",
        width: int = 640,
        height: int = 480,
        fps: float = 30.0,
        max_width: int = 1280,
        max_height: int = 720,
        parent=None,
    ):
        super().__init__(parent)
        self.source = source
        self.width = width
        self.height = height
        self.fps = fps
        self.max_width = max_width
        self.max_height = max_height

        self._cap: cv2.VideoCapture | None = None
        self._running = False
        self._paused = False
        self._use_mock = False
        self._frame_id = 0
        self._last_fps = 0.0

    def _open_source(self) -> bool:
        """根据 source 打开真实视频源或标记为 mock。"""
        self._use_mock = False
        self._cap = None

        if isinstance(self.source, str) and self.source.lower() == "mock":
            self._use_mock = True
            self.status_changed.emit("模拟视频流")
            return True

        # 统一转成 int（摄像头索引）或 str（文件/URL）
        actual_source: Union[int, str]
        if isinstance(self.source, int) or (isinstance(self.source, str) and self.source.isdigit()):
            actual_source = int(self.source)
        elif isinstance(self.source, str) and Path(self.source).exists():
            actual_source = self.source
        elif isinstance(self.source, str) and self.source.startswith(("rtsp://", "http://", "https://")):
            actual_source = self.source
        else:
            self.status_changed.emit(f"错误：源不存在或格式不支持: {self.source}")
            return False

        self._cap = cv2.VideoCapture(actual_source)
        if not self._cap.isOpened():
            self._cap.release()
            self._cap = None
            self.status_changed.emit(f"错误：无法打开视频源: {self.source}")
            return False

        # 限制最大分辨率，避免解码和传输开销过大
        raw_width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or self.width
        raw_height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or self.height
        scale = min(1.0, self.max_width / raw_width, self.max_height / raw_height)
        self.width = int(raw_width * scale)
        self.height = int(raw_height * scale)

        if scale < 1.0:
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        self.fps = self._cap.get(cv2.CAP_PROP_FPS) or self.fps
        self.status_changed.emit(f"已连接: {self.source} ({self.width}×{self.height})")
        return True

    def _read_frame(self) -> np.ndarray | None:
        """读取一帧，mock 模式下生成随机彩色帧。"""
        if self._use_mock:
            image = np.random.randint(0, 255, (self.height, self.width, 3), dtype=np.uint8)
            # 让画面偏暖色调，模拟有人活动的场景
            image[:, :, 2] = np.clip(image[:, :, 2].astype(int) + 40, 0, 255).astype(np.uint8)
            return image

        if self._cap is None:
            return None

        ret, frame = self._cap.read()
        return frame if ret else None

    def _to_pixmap(self, frame: np.ndarray) -> QPixmap:
        """将 OpenCV BGR 帧转成 QPixmap。"""
        # 在这里接入分割模型



        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, channels = rgb.shape
        bytes_per_line = channels * width
        q_image = QImage(rgb.data, width, height, bytes_per_line, QImage.Format_RGB888) # 将opencv的格式转接到QT上面
        return QPixmap.fromImage(q_image)

    def run(self):
        """线程主循环。"""
        self._running = True
        if not self._open_source():
            self._running = False
            return

        target_interval = 1.0 / self.fps
        fps_start = time.perf_counter()
        fps_count = 0

        while self._running:
            if self._paused:
                time.sleep(0.05)
                continue

            loop_start = time.perf_counter()

            frame = self._read_frame()
            # 意外中断
            if frame is None:
                if not self._use_mock:
                    self.status_changed.emit(f"错误：源中断: {self.source}")
                    self._running = False
                    break
                continue

            self._frame_id += 1
            pixmap = self._to_pixmap(frame)
            self.frame_captured.emit(pixmap)
            fps_count += 1

            # FPS 统计：每秒更新一次
            now = time.perf_counter()
            if now - fps_start >= 1.0:
                self._last_fps = fps_count / (now - fps_start)
                self.fps_updated.emit(self._last_fps)
                fps_start = now
                fps_count = 0

            # 动态 sleep，扣除本帧处理耗时，避免累积延迟
            elapsed = time.perf_counter() - loop_start
            sleep_time = max(0.0, target_interval - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)

        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def pause(self):
        self._paused = True
        self.status_changed.emit("已暂停")

    def resume(self):
        self._paused = False
        self.status_changed.emit("播放中")

    def is_paused(self) -> bool:
        return self._paused

    def stop(self):
        self._running = False
        self.wait(1000)

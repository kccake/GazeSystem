"""视频时间轴: 播放控制 + 帧滑块 + 标记条 + 传播进度。

播放节奏由页面驱动(本部件只发 play_toggled, 不内置定时器)。
"""

from __future__ import annotations

from typing import Dict, Optional, Set

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import (QHBoxLayout, QLabel, QProgressBar, QPushButton,
                               QSlider, QVBoxLayout, QWidget)

# 标记颜色: 关键帧=绿, 有提示帧=黄, 复用帧=暗灰
_COLOR_KEYFRAME = QColor("#3fca3f")
_COLOR_PROMPT = QColor("#e5c07b")
_COLOR_REUSED = QColor("#555555")


class _MarkerBar(QWidget):
    """滑块下方的标记条(关键帧=绿色刻度, 有提示帧=黄色刻度, 复用帧=暗灰刻度)。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(10)
        self._num_frames = 0
        self._keyframes: Set[int] = set()
        self._prompt_frames: Set[int] = set()
        self._reused: Set[int] = set()

    def set_state(self, num_frames: int, keyframes: Set[int],
                  prompt_frames: Set[int], reused: Set[int]) -> None:
        self._num_frames = num_frames
        self._keyframes = set(keyframes)
        self._prompt_frames = set(prompt_frames)
        self._reused = set(reused)
        self.update()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#252526"))
        if self._num_frames > 0:
            denom = max(1, self._num_frames - 1)
            w = self.width()
            # 复用帧 -> 提示帧 -> 关键帧, 后者覆盖前者
            for color, frames in ((_COLOR_REUSED, self._reused),
                                  (_COLOR_PROMPT, self._prompt_frames),
                                  (_COLOR_KEYFRAME, self._keyframes)):
                painter.setPen(QPen(color, 2))
                for f in frames:
                    if 0 <= f < self._num_frames:
                        x = int(f / denom * (w - 1))
                        painter.drawLine(x, 1, x, self.height() - 2)
        painter.end()


class TimelineWidget(QWidget):
    """视频时间轴(播放/步进/滑块/标记/进度), 播放节奏由页面驱动。"""

    frame_changed = Signal(int)   # 用户拖动/步进
    play_toggled = Signal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._num_frames = 0
        self._current = 0
        self._playing = False
        self._keyframes: Set[int] = set()
        self._prompt_frames: Set[int] = set()
        self._reused: Dict[int, int] = {}
        self._setup_ui()
        self._refresh_enabled()

    # ---------------- 公共 API ----------------
    def set_range(self, num_frames: int) -> None:
        """设置总帧数。"""
        self._num_frames = max(0, int(num_frames))
        self.slider.setRange(0, max(0, self._num_frames - 1))
        if self._current > max(0, self._num_frames - 1):
            self._current = max(0, self._num_frames - 1)
        self._sync_slider()
        self._sync_markers()
        self._refresh_enabled()

    def set_current(self, i: int, emit: bool = False) -> None:
        """设置当前帧(默认不发 frame_changed 避免递归; emit=True 时发)。"""
        i = max(0, min(int(i), max(0, self._num_frames - 1)))
        changed = i != self._current
        self._current = i
        self._sync_slider()
        if emit and (changed or self._num_frames > 0):
            self.frame_changed.emit(i)

    def current(self) -> int:
        return self._current

    def set_markers(self, keyframes: Optional[Set[int]] = None,
                    prompt_frames: Optional[Set[int]] = None,
                    reused: Optional[Dict[int, int]] = None) -> None:
        """设置标记: 关键帧(绿) / 有提示帧(黄) / 复用帧(暗灰, 传 {帧: 源关键帧} 字典)。"""
        self._keyframes = set(keyframes) if keyframes else set()
        self._prompt_frames = set(prompt_frames) if prompt_frames else set()
        self._reused = dict(reused) if reused else {}
        self._sync_markers()

    def clear_markers(self) -> None:
        """清空全部标记。"""
        self.set_markers()

    def set_progress(self, p: Optional[float]) -> None:
        """传播进度 0~1; None 隐藏进度条。"""
        if p is None:
            self.progress.hide()
            return
        self.progress.show()
        self.progress.setValue(int(max(0.0, min(1.0, p)) * 100))

    def set_playing(self, playing: bool) -> None:
        """同步播放按钮状态(不发 play_toggled)。"""
        self._playing = bool(playing)
        self.play_btn.setText("⏸" if self._playing else "▶")

    # ---------------- 内部 ----------------
    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 2, 4, 2)
        layout.setSpacing(2)

        row = QHBoxLayout()
        row.setSpacing(6)
        self.play_btn = QPushButton("▶")
        self.play_btn.setFixedWidth(48)
        self.prev_btn = QPushButton("⏮")
        self.prev_btn.setFixedWidth(36)
        self.next_btn = QPushButton("⏭")
        self.next_btn.setFixedWidth(36)
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, 0)
        self.frame_label = QLabel("--/--")
        self.frame_label.setMinimumWidth(70)
        self.frame_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        row.addWidget(self.play_btn)
        row.addWidget(self.prev_btn)
        row.addWidget(self.next_btn)
        row.addWidget(self.slider, stretch=1)
        row.addWidget(self.frame_label)
        layout.addLayout(row)

        self.marker_bar = _MarkerBar()
        layout.addWidget(self.marker_bar)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setFixedHeight(14)
        self.progress.hide()
        layout.addWidget(self.progress)

        self.play_btn.clicked.connect(self._on_play_clicked)
        self.prev_btn.clicked.connect(lambda: self._step(-1))
        self.next_btn.clicked.connect(lambda: self._step(1))
        self.slider.valueChanged.connect(self._on_slider_changed)

    def _on_play_clicked(self) -> None:
        self.set_playing(not self._playing)
        self.play_toggled.emit(self._playing)

    def _step(self, delta: int) -> None:
        if self._num_frames <= 0:
            return
        target = max(0, min(self._current + delta, self._num_frames - 1))
        if target != self._current:
            self.set_current(target, emit=True)

    def _on_slider_changed(self, value: int) -> None:
        # 仅用户拖动触发(_sync_slider 已屏蔽程序设置)
        self._current = value
        self._update_label()
        self.frame_changed.emit(value)

    def _sync_slider(self) -> None:
        self.slider.blockSignals(True)
        self.slider.setValue(self._current)
        self.slider.blockSignals(False)
        self._update_label()

    def _sync_markers(self) -> None:
        self.marker_bar.set_state(self._num_frames, self._keyframes,
                                  self._prompt_frames, set(self._reused.keys()))

    def _update_label(self) -> None:
        if self._num_frames > 0:
            self.frame_label.setText(f"{self._current}/{self._num_frames - 1}")
        else:
            self.frame_label.setText("--/--")

    def _refresh_enabled(self) -> None:
        ok = self._num_frames > 0
        self.play_btn.setEnabled(ok)
        self.prev_btn.setEnabled(ok)
        self.next_btn.setEnabled(ok)
        self.slider.setEnabled(ok)

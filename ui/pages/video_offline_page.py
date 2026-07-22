"""视频离线标注页: 上传整段视频 -> 本地解码显示 -> 批量标注(auto_predict=False)
-> 提交传播 -> 逐帧查看 / 导出掩码。

数据流:
- 上传: run_api(create_offline_session) 拿到 session_id/num_frames 后,
  用 cv2 打开同一文件做本地解码显示, VideoSessionWorker 驱动 WS 会话。
- 提示: 画布交互 -> 本地乐观记录 self._prompts + 入队服务端命令;
  worker 串行执行, 每条命令恰好对应一个 command_result 或一次 failed,
  与本地 _journal(FIFO) 一一对应: 确认即出队, 失败则出队并回滚本地状态。
- 传播: worker.submit() 事件流, keyframe 事件带掩码 -> self._mask_cache。
- 取帧: submit 完成后 seek 到未缓存帧 -> get_frame 节流拉取
  (同一时刻只允许一个在途, 期间的新 seek 只记最新目标)。

注意(服务端语义, 与 business/service_layer.py 对齐, 仅引用不 import):
- auto_predict=False 时 add_point/add_box 只记账, submit 才计算。
- reset_video_tracking 会清空全部提示(frame_prompts.clear), 本地 _prompts 同步清空。
- load_video_prompt_file 的 merge_mode: append 同帧同组叠加(框覆盖) /
  replace 清该帧该组再载 / skip 同帧同组已存在则跳过。
"""

from __future__ import annotations

import os
import time
from collections import deque
from typing import Callable, Deque, Dict, List, Optional, Set, Tuple

import cv2
import numpy as np
from PIL import Image
from PySide6.QtCore import Qt, QThread, QTimer, Signal
from PySide6.QtGui import QImage, QKeySequence, QShortcut
from PySide6.QtWidgets import (QFileDialog, QHBoxLayout, QLabel, QLineEdit,
                               QMessageBox, QProgressDialog, QPushButton,
                               QSpinBox, QVBoxLayout, QWidget)

from ..context import AppContext
from ..prompt_file import load_file, save_video_file
from ..utils import pil_to_qimage
from ..widgets.canvas import AnnotationCanvas
from ..widgets.group_panel import GroupPanel
from ..widgets.timeline import TimelineWidget
from ..workers import VideoSessionWorker, run_api

# 一个提示条目: points=[(x, y, label), ...], box=(x1, y1, x2, y2) 或 None
_PromptEntry = Dict[str, object]
# 可选择视频文件的过滤器
_VIDEO_FILTER = "视频文件 (*.mp4 *.avi *.mkv *.mov)"


class _VideoExportThread(QThread):
    """分割视频导出线程: 掩码叠加回原视频写出 mp4(纯本地合成, 不走网络)。

    掩码来源是页面 _mask_cache 的浅拷快照(PIL 不可变, 线程间共享安全);
    非关键帧复用最近前序已缓存帧, 与页面显示/服务端插帧同一语义。
    """

    progressed = Signal(int, int)   # 已写入帧数, 总帧数
    failed = Signal(str)
    done = Signal(bool, str)        # (是否完整导出, 输出路径)

    def __init__(self, video_path: str, out_path: str, fps: float,
                 num_frames: int, mask_cache: Dict[int, Dict[int, Image.Image]],
                 colors_bgr: Dict[int, Tuple[int, int, int]],
                 opacity: int = 128, parent=None):
        super().__init__(parent)
        self._video_path = video_path
        self._out_path = out_path
        self._fps = fps if fps and fps > 1e-3 else 25.0
        self._num_frames = num_frames
        self._mask_cache = mask_cache
        self._colors_bgr = colors_bgr
        self._alpha = max(0, min(255, opacity)) / 255.0
        self._cancel = False

    def cancel(self) -> None:
        self._cancel = True

    def run(self) -> None:
        try:
            cap = cv2.VideoCapture(self._video_path)
            if not cap.isOpened():
                self.failed.emit(f"无法打开源视频: {self._video_path}")
                return
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            writer = cv2.VideoWriter(self._out_path,
                                     cv2.VideoWriter_fourcc(*"mp4v"),
                                     self._fps, (w, h))
            if not writer.isOpened():
                cap.release()
                self.failed.emit(f"无法创建输出文件: {self._out_path}")
                return
            keys = sorted(self._mask_cache.keys())
            ki = -1                    # keys 中 <= 当前帧 的最后下标
            cur_masks = None           # 当前复用的 {gid: np 掩码}
            cancelled = False
            for i in range(self._num_frames):
                if self._cancel:
                    cancelled = True
                    break
                ok, frame = cap.read()
                if not ok or frame is None:
                    break              # 读不到就按已完成部分收尾
                while ki + 1 < len(keys) and keys[ki + 1] <= i:
                    ki += 1
                    cur_masks = {g: np.asarray(m)
                                 for g, m in self._mask_cache[keys[ki]].items()}
                if cur_masks:
                    overlay = frame.copy()
                    for g, m in cur_masks.items():
                        color = self._colors_bgr.get(g)
                        if color is None:
                            continue   # 隐藏组不导出
                        overlay[m > 0] = color
                    frame = cv2.addWeighted(overlay, self._alpha,
                                            frame, 1.0 - self._alpha, 0)
                writer.write(frame)
                self.progressed.emit(i + 1, self._num_frames)
            writer.release()
            cap.release()
            self.done.emit(not cancelled, self._out_path)
        except Exception as e:  # 导出线程内任何异常都回投 UI
            self.failed.emit(str(e))


class VideoOfflinePage(QWidget):
    """视频离线标注页(批量标注 + 整段提交传播)。"""

    def __init__(self, context: AppContext, parent=None):
        super().__init__(parent)
        self._ctx = context

        # ---- 会话状态 ----
        self._session_id: Optional[str] = None
        self._worker: Optional[VideoSessionWorker] = None
        self._video_path: Optional[str] = None
        self._cap: Optional[cv2.VideoCapture] = None
        self._cap_pos: Optional[int] = None  # 解码器下一帧序号(顺序读免 seek)
        self._num_frames: int = 0
        self._fps: float = 25.0
        self._uploading: bool = False
        self._broken: bool = False          # WS 连接意外断开
        self._expected_close: bool = False  # 区分主动关闭与意外断开

        # ---- 播放状态 ----
        self._current_frame: int = 0
        self._playing: bool = False
        self._play_timer = QTimer(self)
        self._play_timer.timeout.connect(self._advance_frame)

        # ---- 标注数据(页面是唯一数据源, 画布只负责显示) ----
        # frame_idx -> gid -> {"points": [(x,y,label)], "box": tuple|None}
        self._prompts: Dict[int, Dict[int, _PromptEntry]] = {}
        self._dirty: int = 0  # 未提交的提示操作数(本地计数, prompts_applied 后清零)

        # ---- 结果缓存与标记 ----
        self._mask_cache: Dict[int, Dict[int, Image.Image]] = {}
        self._keyframes: Set[int] = set()
        self._reused: Dict[int, int] = {}   # 复用帧 -> 源关键帧
        self._submitted_once: bool = False  # submit 完成过一次后可 seek 取帧/导出掩码

        # ---- 命令配对(FIFO) ----
        self._journal: Deque[dict] = deque()

        # ---- 传播状态 ----
        self._propagating: bool = False
        self._submit_pending: bool = False  # 已点提交, 等 propagate_start
        self._num_keyframes: int = 0
        self._keyframe_count: int = 0

        # ---- get_frame 节流 ----
        self._gf_inflight: bool = False
        self._gf_pending: Optional[int] = None

        # ---- prompt 文件导入 ----
        self._pending_import: Optional[Tuple[dict, str]] = None

        # ---- mask 导出 ----
        self._exporting: bool = False
        self._export_video_thread: Optional[_VideoExportThread] = None
        self._export_stop: bool = False
        self._export_state: Optional[dict] = None
        self._export_dialog: Optional[QProgressDialog] = None

        self._setup_ui()
        self._setup_shortcuts()
        self._refresh_ui_state()

    # ================= UI 构建 =================
    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)

        # ---- 顶部: 会话控制条 ----
        top = QHBoxLayout()
        top.setSpacing(6)
        self.choose_btn = QPushButton("选择视频…")
        self.choose_btn.clicked.connect(self._on_choose_video)
        top.addWidget(self.choose_btn)
        self.path_edit = QLineEdit()
        self.path_edit.setReadOnly(True)
        self.path_edit.setPlaceholderText("未选择视频文件")
        top.addWidget(self.path_edit, stretch=1)
        top.addWidget(QLabel("帧间隔:"))
        self.stride_spin = QSpinBox()
        self.stride_spin.setRange(1, 100)
        self.stride_spin.setValue(1)
        self.stride_spin.setToolTip("frame_stride: 每隔多少帧取一个关键帧")
        top.addWidget(self.stride_spin)
        self.upload_btn = QPushButton("上传建会话")
        self.upload_btn.clicked.connect(self._on_upload)
        top.addWidget(self.upload_btn)
        self.close_btn = QPushButton("关闭会话")
        self.close_btn.clicked.connect(self._on_close_clicked)
        top.addWidget(self.close_btn)
        layout.addLayout(top)

        # ---- 中部: 左组面板 + 右(标注工具条 + 画布) ----
        mid = QHBoxLayout()
        mid.setSpacing(4)
        self.group_panel = GroupPanel()
        self.group_panel.setFixedWidth(220)
        self.group_panel.group_removed.connect(self._on_group_removed)
        self.group_panel.visibility_changed.connect(self._on_visibility_changed)
        mid.addWidget(self.group_panel)

        right = QVBoxLayout()
        right.setSpacing(4)
        tool = QHBoxLayout()
        tool.setSpacing(6)
        self.point_mode_btn = QPushButton("点模式 (P)")
        self.point_mode_btn.setCheckable(True)
        self.point_mode_btn.setChecked(True)
        self.point_mode_btn.clicked.connect(lambda: self._set_mode("point"))
        tool.addWidget(self.point_mode_btn)
        self.box_mode_btn = QPushButton("框模式 (B)")
        self.box_mode_btn.setCheckable(True)
        self.box_mode_btn.clicked.connect(lambda: self._set_mode("box"))
        tool.addWidget(self.box_mode_btn)
        hint = QLabel("左键正点 / 右键负点")
        hint.setStyleSheet("color: #888888;")
        tool.addWidget(hint)
        tool.addStretch(1)
        self.undo_point_btn = QPushButton("撤销点")
        self.undo_point_btn.setToolTip("删除当前帧当前组的最后一个点")
        self.undo_point_btn.clicked.connect(self._on_undo_point_clicked)
        tool.addWidget(self.undo_point_btn)
        self.clear_box_btn = QPushButton("清除框")
        self.clear_box_btn.setToolTip("清除当前帧当前组的框")
        self.clear_box_btn.clicked.connect(self._on_clear_box_clicked)
        tool.addWidget(self.clear_box_btn)
        self.clear_group_btn = QPushButton("清除组")
        self.clear_group_btn.setToolTip("清除当前组在全部帧上的提示")
        self.clear_group_btn.clicked.connect(self._on_clear_group_clicked)
        tool.addWidget(self.clear_group_btn)
        right.addLayout(tool)

        self.canvas = AnnotationCanvas()
        self.canvas.point_added.connect(self._on_point_added)
        self.canvas.box_drawn.connect(self._on_box_drawn)
        right.addWidget(self.canvas, stretch=1)
        mid.addLayout(right, stretch=1)
        layout.addLayout(mid, stretch=1)

        # ---- 底部: 时间轴 ----
        self.timeline = TimelineWidget()
        self.timeline.frame_changed.connect(self._on_frame_changed)
        self.timeline.play_toggled.connect(self._on_play_toggled)
        layout.addWidget(self.timeline)

        # ---- 底部: 传播控制行 ----
        row1 = QHBoxLayout()
        row1.setSpacing(6)
        self.submit_btn = QPushButton("提交传播")
        self.submit_btn.clicked.connect(self._on_submit_clicked)
        row1.addWidget(self.submit_btn)
        self.cancel_btn = QPushButton("取消")
        self.cancel_btn.clicked.connect(self._on_cancel_clicked)
        row1.addWidget(self.cancel_btn)
        self.reset_btn = QPushButton("重置追踪")
        self.reset_btn.clicked.connect(self._on_reset_clicked)
        row1.addWidget(self.reset_btn)
        self.dirty_label = QLabel("待提交提示: 0")
        row1.addWidget(self.dirty_label)
        self.progress_label = QLabel("")
        self.progress_label.setStyleSheet("color: #aaaaaa;")
        row1.addWidget(self.progress_label, stretch=1)
        layout.addLayout(row1)

        # ---- 底部: 文件/导出行 + 右下角状态标签 ----
        row2 = QHBoxLayout()
        row2.setSpacing(6)
        self.import_btn = QPushButton("导入提示文件")
        self.import_btn.clicked.connect(self._on_import_clicked)
        row2.addWidget(self.import_btn)
        self.export_prompt_btn = QPushButton("导出提示文件")
        self.export_prompt_btn.clicked.connect(self._on_export_prompt_clicked)
        row2.addWidget(self.export_prompt_btn)
        self.export_mask_btn = QPushButton("导出 Mask 序列")
        self.export_mask_btn.clicked.connect(self._on_export_masks_clicked)
        row2.addWidget(self.export_mask_btn)
        self.export_video_btn = QPushButton("导出分割视频")
        self.export_video_btn.setToolTip("把掩码叠加回原视频导出 mp4(本地合成)")
        self.export_video_btn.clicked.connect(self._on_export_video_clicked)
        row2.addWidget(self.export_video_btn)
        row2.addStretch(1)
        self.status_label = QLabel("请选择视频文件并上传建立会话")
        self.status_label.setStyleSheet("color: #cccccc;")
        row2.addWidget(self.status_label)
        layout.addLayout(row2)

        # 按钮不抢焦点, 让空格/方向键快捷键稳定生效
        for btn in self.findChildren(QPushButton):
            btn.setFocusPolicy(Qt.NoFocus)

    def _setup_shortcuts(self) -> None:
        """空格=播放/暂停, ←/→=上/下一帧, P=点模式, B=框模式。

        WidgetWithChildrenShortcut: 只在本页聚焦时生效;
        焦点在文本框(组重命名)时按键仍归文本框(ShortcutOverride)。
        """
        def _sc(key, slot):
            sc = QShortcut(QKeySequence(key), self)
            sc.setContext(Qt.WidgetWithChildrenShortcut)
            sc.activated.connect(slot)

        _sc(Qt.Key_Space, self._toggle_play)
        _sc(Qt.Key_Left, lambda: self._step_frame(-1))
        _sc(Qt.Key_Right, lambda: self._step_frame(1))
        _sc(Qt.Key_P, lambda: self._set_mode("point"))
        _sc(Qt.Key_B, lambda: self._set_mode("box"))

    # ================= 状态辅助 =================
    def _set_status(self, text: str, bar: bool = True) -> None:
        """右下角状态标签 + (可选)主窗口状态栏。"""
        self.status_label.setText(text)
        if bar:
            self._ctx.status_message.emit(text)

    def _refresh_ui_state(self) -> None:
        """根据会话/传播/导出等标志统一刷新各控件可用态。"""
        has = self._session_id is not None
        busy = self._propagating or self._submit_pending
        interactive = has and not self._broken and not self._exporting
        can_annotate = interactive and not busy
        idle_journal = not self._journal

        self.choose_btn.setEnabled(not has and not self._uploading)
        self.path_edit.setEnabled(not has and not self._uploading)
        self.stride_spin.setEnabled(not has and not self._uploading)
        self.upload_btn.setEnabled(not has and not self._uploading
                                   and bool(self._video_path))
        self.close_btn.setEnabled(has)

        self.canvas.setEnabled(can_annotate)
        self.group_panel.setEnabled(can_annotate)
        for b in (self.point_mode_btn, self.box_mode_btn,
                  self.undo_point_btn, self.clear_box_btn, self.clear_group_btn):
            b.setEnabled(can_annotate)
        self.timeline.setEnabled(interactive and self._num_frames > 0)

        # submit 要求命令队列已清空: 保证 failed 能无歧义地区分
        # "提交前的命令失败" 与 "submit 本身失败"(详见 _on_worker_failed)
        self.submit_btn.setEnabled(can_annotate and idle_journal
                                   and bool(self._prompts))
        self.cancel_btn.setEnabled(self._propagating)
        self.reset_btn.setEnabled(can_annotate and idle_journal)
        self.import_btn.setEnabled(can_annotate and self._pending_import is None)
        self.export_prompt_btn.setEnabled(interactive and bool(self._prompts))
        self.export_mask_btn.setEnabled(can_annotate and idle_journal
                                        and self._submitted_once)
        self.export_video_btn.setEnabled(interactive and self._submitted_once
                                         and bool(self._mask_cache))

    # ================= 会话管理 =================
    def _on_choose_video(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "选择视频文件", "", _VIDEO_FILTER)
        if not path:
            return
        self._video_path = path
        self.path_edit.setText(path)
        self.path_edit.setToolTip(path)
        self._refresh_ui_state()

    def _on_upload(self) -> None:
        if not self._video_path or self._session_id is not None or self._uploading:
            return
        self._uploading = True
        self.upload_btn.setText("上传中…")
        self._set_status("正在上传视频并创建会话(大文件可能需要较长时间)…")
        self._refresh_ui_state()
        run_api(self._ctx.client.create_offline_session,
                self._video_path, int(self.stride_spin.value()), False,
                on_ok=self._on_session_created,
                on_err=self._on_session_failed, parent=self)

    def _on_session_created(self, result: dict) -> None:
        self._uploading = False
        self.upload_btn.setText("上传建会话")
        sid = result.get("session_id")
        num_frames = int(result.get("num_frames", 0))
        if not sid or num_frames <= 0:
            self._set_status(f"建会话失败: 返回异常 {result}")
            self._refresh_ui_state()
            return

        # 本地解码显示用(服务端只回了帧数, 画面由 cv2 自行解码)
        cap = cv2.VideoCapture(self._video_path)
        if not cap.isOpened():
            cap.release()
            run_api(self._ctx.client.close_video_session, sid)  # 收拾服务端会话
            QMessageBox.warning(self, "打开失败", "cv2 无法解码该视频文件")
            self._set_status("本地解码失败, 已放弃会话")
            self._refresh_ui_state()
            return
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps != fps or fps <= 1e-3:  # 0 / NaN 兜底
            fps = 25.0

        self._session_id = sid
        self._num_frames = num_frames
        self._fps = float(fps)
        self._cap = cap
        self._cap_pos = None
        self._current_frame = 0
        self._broken = False
        self._expected_close = False

        worker = VideoSessionWorker(self._ctx.client, sid, parent=self)
        worker.command_result.connect(self._on_command_result)
        worker.propagate_event.connect(self._on_propagate_event)
        worker.failed.connect(self._on_worker_failed)
        worker.closed.connect(self._on_worker_closed)
        self._worker = worker
        worker.start()

        self.timeline.set_range(num_frames)
        self._on_frame_changed(0)
        self._set_status(f"会话已建立: {num_frames} 帧, {self._fps:.1f} fps")
        self._refresh_ui_state()

    def _on_session_failed(self, msg: str) -> None:
        self._uploading = False
        self.upload_btn.setText("上传建会话")
        QMessageBox.warning(self, "上传失败", f"创建离线会话失败:\n{msg}")
        self._set_status(f"上传失败: {msg}")
        self._refresh_ui_state()

    def _on_close_clicked(self) -> None:
        self._close_session()
        self._set_status("会话已关闭")

    def _close_session(self) -> None:
        """关闭会话并复位全部页面状态(幂等)。"""
        if self._session_id is None and self._worker is None:
            return
        self._play_timer.stop()
        self._playing = False
        self.timeline.set_playing(False)

        self._expected_close = True
        worker, self._worker = self._worker, None
        if worker is not None:
            worker.cancel()    # 停掉在途传播(通道随后关闭, 串话无碍)
            worker.shutdown()
            if not worker.wait(5000):  # 等不及则脱钩, 避免 QThread 运行中被析构
                worker.setParent(None)
                worker.finished.connect(worker.deleteLater)

        sid, self._session_id = self._session_id, None
        if sid is not None:
            # fire-and-forget: 页面可能随后即析构, worker 不挂本页父子关系
            run_api(self._ctx.client.close_video_session, sid,
                    on_err=lambda msg: self._ctx.status_message.emit(
                        f"关闭会话失败: {msg}"))

        if self._cap is not None:
            self._cap.release()
            self._cap = None

        # 状态复位
        self._num_frames = 0
        self._current_frame = 0
        self._prompts.clear()
        self._mask_cache.clear()
        self._keyframes.clear()
        self._reused.clear()
        self._journal.clear()
        self._dirty = 0
        self._submitted_once = False
        self._propagating = self._submit_pending = False
        self._broken = False
        self._gf_inflight = False
        self._gf_pending = None
        self._pending_import = None
        self._exporting = False
        self._export_state = None

        self.group_panel.clear()
        self.canvas.clear_masks()
        self.canvas.set_prompts([], [])
        self.timeline.set_range(0)
        self.timeline.clear_markers()
        self.timeline.set_progress(None)
        self._refresh_dirty()
        self.progress_label.setText("")
        self._refresh_ui_state()

    # ================= 播放与帧显示 =================
    def _can_play(self) -> bool:
        return (self._session_id is not None and self._cap is not None
                and self._num_frames > 0 and not self._broken
                and not self._exporting)

    def _on_play_toggled(self, playing: bool) -> None:
        self._playing = playing
        if playing:
            if not self._can_play():
                self._playing = False
                self.timeline.set_playing(False)
                return
            # 在末帧按播放 -> 从头再来
            if self._current_frame >= self._num_frames - 1:
                self.timeline.set_current(0, emit=True)
            self._play_timer.start(max(10, int(1000.0 / self._fps)))
        else:
            self._play_timer.stop()

    def _toggle_play(self) -> None:
        if not self._can_play():
            return
        playing = not self._playing
        self.timeline.set_playing(playing)  # 回同步按钮(不再发 play_toggled)
        self._on_play_toggled(playing)

    def _advance_frame(self) -> None:
        nxt = self._current_frame + 1
        if nxt >= self._num_frames:  # 播到末尾自动停
            self._play_timer.stop()
            self._playing = False
            self.timeline.set_playing(False)
            return
        self.timeline.set_current(nxt, emit=True)

    def _step_frame(self, delta: int) -> None:
        if self._session_id is None or self._num_frames <= 0:
            return
        self.timeline.set_current(self._current_frame + delta, emit=True)

    def _read_frame(self, idx: int) -> Optional[QImage]:
        """cv2 解码第 idx 帧 -> QImage(RGB888, 独立内存)。
        顺序读(播放场景)直接 read 不走 seek; 只有随机跳转才 seek。
        逐帧 seek(H.264 需回退关键帧重解码)是播放卡顿的主因。"""
        cap = self._cap
        if cap is None:
            return None
        if idx != self._cap_pos:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok or frame is None:
            self._cap_pos = None  # 位置失知, 下次强制 seek
            return None
        self._cap_pos = idx + 1
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        return QImage(rgb.data, w, h, w * 3, QImage.Format_RGB888).copy()

    def _on_frame_changed(self, idx: int) -> None:
        """换帧总入口: 解码显示 -> 重建提示渲染 -> 刷新掩码渲染。"""
        if self._cap is None or self._num_frames <= 0:
            return
        idx = max(0, min(idx, self._num_frames - 1))
        self._current_frame = idx
        img = self._read_frame(idx)
        if img is not None:
            self.canvas.set_image(img)
        else:
            self._set_status(f"第 {idx} 帧本地解码失败", bar=False)
        self._refresh_prompt_view()
        self._show_masks_for(idx)

    # ================= 标注(批量模式, auto_predict=False) =================
    def _annot_ready(self) -> bool:
        return (self._worker is not None and self._session_id is not None
                and not self._broken and not self._propagating
                and not self._submit_pending and not self._exporting)

    def _set_mode(self, mode: str) -> None:
        self.canvas.set_mode(mode)
        self.point_mode_btn.setChecked(mode == "point")
        self.box_mode_btn.setChecked(mode == "box")

    def _current_gid(self) -> int:
        """当前组; 无组时自动新建一个。"""
        gid = self.group_panel.current_group()
        if gid is None:
            gid = self.group_panel.add_group()
            self._sync_group_colors()
        return gid

    def _entry(self, frame: int, gid: int) -> _PromptEntry:
        return self._prompts.setdefault(frame, {}).setdefault(
            gid, {"points": [], "box": None})

    def _prune(self, frame: int, gid: int) -> None:
        """清掉空条目(无点无框), 保持与服务端 frame_prompts 清理口径一致。"""
        frame_groups = self._prompts.get(frame)
        if not frame_groups:
            return
        entry = frame_groups.get(gid)
        if entry is not None and not entry["points"] and entry["box"] is None:
            del frame_groups[gid]
        if not frame_groups:
            del self._prompts[frame]

    def _on_point_added(self, x: float, y: float, label: int) -> None:
        if not self._annot_ready():
            return
        frame, gid = self._current_frame, self._current_gid()
        pt = (x, y, label)
        self._entry(frame, gid)["points"].append(pt)
        self._bump_dirty(+1)
        self._journal.append({
            "action": "add_point",
            "undo": lambda: self._undo_add_point(frame, gid, pt)})
        self._worker.add_point(gid, x, y, label, frame)
        self._invalidate_cache_from(frame)
        self._after_prompts_changed()

    def _on_box_drawn(self, x1: float, y1: float, x2: float, y2: float) -> None:
        if not self._annot_ready():
            return
        frame, gid = self._current_frame, self._current_gid()
        entry = self._entry(frame, gid)
        old_box = entry["box"]
        entry["box"] = (x1, y1, x2, y2)
        self._bump_dirty(+1)
        self._journal.append({
            "action": "add_box",
            "undo": lambda: self._undo_set_box(frame, gid, old_box)})
        self._worker.add_box(gid, x1, y1, x2, y2, frame)
        self._invalidate_cache_from(frame)
        self._after_prompts_changed()

    def _on_undo_point_clicked(self) -> None:
        if not self._annot_ready():
            return
        gid = self.group_panel.current_group()
        if gid is None:
            self._set_status("请先选择一个组")
            return
        frame = self._current_frame
        entry = self._prompts.get(frame, {}).get(gid)
        if not entry or not entry["points"]:
            self._set_status("当前帧当前组没有可撤销的点")
            return
        idx = len(entry["points"]) - 1
        pt = entry["points"].pop()
        self._prune(frame, gid)
        self._bump_dirty(+1)
        self._journal.append({
            "action": "delete_point",
            "undo": lambda: self._undo_delete_point(frame, gid, pt, idx)})
        self._worker.delete_point(gid, frame, -1)
        self._invalidate_cache_from(frame)
        self._after_prompts_changed()

    def _on_clear_box_clicked(self) -> None:
        if not self._annot_ready():
            return
        gid = self.group_panel.current_group()
        if gid is None:
            self._set_status("请先选择一个组")
            return
        frame = self._current_frame
        entry = self._prompts.get(frame, {}).get(gid)
        if not entry or entry["box"] is None:
            self._set_status("当前帧当前组没有框")
            return
        old_box = entry["box"]
        entry["box"] = None
        self._prune(frame, gid)
        self._bump_dirty(+1)
        self._journal.append({
            "action": "clear_box",
            "undo": lambda: self._undo_set_box(frame, gid, old_box)})
        self._worker.clear_box(gid, frame)
        self._invalidate_cache_from(frame)
        self._after_prompts_changed()

    def _on_clear_group_clicked(self) -> None:
        if not self._annot_ready():
            return
        gid = self.group_panel.current_group()
        if gid is None:
            self._set_status("请先选择一个组")
            return
        # 走组面板的删除路径, 与点行上的 ✕ 行为一致
        self.group_panel.remove_group(gid)

    def _on_group_removed(self, gid: int) -> None:
        """组被删除(面板 ✕ 或清除组按钮): 清掉该组在全部帧的提示(服务端+本地)。"""
        if not self._annot_ready():
            return
        snapshot = {f: {"points": list(groups[gid]["points"]),
                        "box": groups[gid]["box"]}
                    for f, groups in self._prompts.items() if gid in groups}
        if not snapshot:
            return  # 该组没有任何提示, 纯面板操作
        for f in snapshot:
            del self._prompts[f][gid]
            if not self._prompts[f]:
                del self._prompts[f]
        self._bump_dirty(+1)
        self._journal.append({
            "action": "clear_group",
            "undo": lambda: self._undo_clear_group(gid, snapshot)})
        self._worker.clear_group(gid)
        # 服务端会删该组全部痕迹(含缓存结果中的对应行), 本地缓存镜像
        for masks in self._mask_cache.values():
            masks.pop(gid, None)
        self._after_prompts_changed()
        self._show_masks_for(self._current_frame)

    # ---- 失败回滚(与本地乐观应用互逆) ----
    def _undo_add_point(self, frame: int, gid: int, pt) -> None:
        entry = self._prompts.get(frame, {}).get(gid)
        if entry:
            pts = entry["points"]
            for i in range(len(pts) - 1, -1, -1):
                if pts[i] == pt:
                    pts.pop(i)
                    break
            self._prune(frame, gid)

    def _undo_delete_point(self, frame: int, gid: int, pt, idx: int) -> None:
        entry = self._entry(frame, gid)
        entry["points"].insert(min(idx, len(entry["points"])), pt)

    def _undo_set_box(self, frame: int, gid: int, old_box) -> None:
        entry = self._entry(frame, gid)
        entry["box"] = old_box
        self._prune(frame, gid)

    def _undo_clear_group(self, gid: int, snapshot: Dict[int, _PromptEntry]) -> None:
        for f, data in snapshot.items():
            self._prompts.setdefault(f, {})[gid] = data
        self.group_panel.ensure_group(gid)  # 面板行一并恢复
        self._sync_group_colors()

    # ================= 提示渲染 / 标记 / 计数 =================
    def _after_prompts_changed(self) -> None:
        self._refresh_prompt_view()
        self._refresh_markers()
        self._refresh_ui_state()  # submit 可用态依赖 journal/prompts

    def _refresh_prompt_view(self) -> None:
        """用当前帧的本地提示重建画布 set_prompts。"""
        points: List[Tuple[float, float, int, int]] = []
        boxes: List[Tuple[float, float, float, float, int]] = []
        for gid, entry in self._prompts.get(self._current_frame, {}).items():
            for x, y, lbl in entry["points"]:
                points.append((x, y, lbl, gid))
            if entry["box"] is not None:
                boxes.append((*entry["box"], gid))
        self.canvas.set_prompts(points, boxes)

    def _refresh_markers(self) -> None:
        self.timeline.set_markers(keyframes=self._keyframes,
                                  prompt_frames=set(self._prompts.keys()),
                                  reused=self._reused)

    def _bump_dirty(self, delta: int) -> None:
        self._dirty = max(0, self._dirty + delta)
        self._refresh_dirty()

    def _refresh_dirty(self) -> None:
        self.dirty_label.setText(f"待提交提示: {self._dirty}")

    def _sync_group_colors(self) -> None:
        self.canvas.set_group_colors(
            {g: self.group_panel.color_of(g) for g in self.group_panel.groups()})

    def _on_visibility_changed(self, gid: int, visible: bool) -> None:
        hidden = {g for g in self.group_panel.groups()
                  if not self.group_panel.is_visible(g)}
        self.canvas.set_hidden_groups(hidden)

    # ================= 掩码缓存与显示 =================
    def _latest_cached_frame_before(self, frame: int) -> Optional[int]:
        """不大于 frame 的最近已缓存帧(镜像服务端 _latest_cached_before)。"""
        candidates = [f for f in self._mask_cache if f <= frame]
        return max(candidates) if candidates else None

    def _show_masks_for(self, frame: int) -> None:
        """换帧时刷新掩码: 命中缓存直接显示; 未命中复用最近前序已缓存帧
        (与服务端插帧同语义的本地复用); 暂停且无待提交提示时才拉取权威结果。"""
        cached = self._mask_cache.get(frame)
        if cached is not None:
            self.canvas.set_masks({g: pil_to_qimage(m) for g, m in cached.items()})
            return
        prev = self._latest_cached_frame_before(frame)
        if prev is not None:
            # 非网格帧: 复用最近前序帧掩码(与服务端 reused_from 语义一致)
            self.canvas.set_masks({g: pil_to_qimage(m)
                                   for g, m in self._mask_cache[prev].items()})
            self._set_status(f"插帧: 复用第 {prev} 帧掩码", bar=False)
        else:
            self.canvas.clear_masks()
        # 播放中靠本地复用即时显示; 暂停且提示已全部提交时才向服务端拉取该帧权威结果
        if (self._submitted_once and self._dirty == 0
                and self._annot_ready()
                and not self._play_timer.isActive()):
            self._request_frame_result(frame)

    def _invalidate_cache_from(self, frame: int) -> None:
        """提示变化后丢弃 frame 及之后的掩码缓存(镜像服务端 _invalidate_from)。"""
        for f in [f for f in self._mask_cache if f >= frame]:
            del self._mask_cache[f]
        self._keyframes = {f for f in self._keyframes if f < frame}
        self._reused = {f: src for f, src in self._reused.items() if f < frame}
        self._refresh_markers()
        self._show_masks_for(self._current_frame)  # 当前帧显示可能已失效, 重渲染

    # ================= get_frame 节流 =================
    def _request_frame_result(self, frame: int) -> None:
        """单在途节流: 有在途请求时只记最新目标, 结果回来再补发。"""
        if self._gf_inflight:
            self._gf_pending = frame
            return
        self._gf_inflight = True
        self._journal.append({"action": "get_frame", "undo": None, "meta": frame})
        self._worker.get_frame(frame, False)

    def _on_frame_result(self, ev: dict) -> None:
        self._gf_inflight = False
        frame = int(ev.get("frame_idx", -1))
        masks = ev.get("mask_images")
        self._mask_cache[frame] = dict(masks) if masks else {}
        # markers 按结果里的 keyframe/reused_from 更新
        if ev.get("keyframe"):
            self._keyframes.add(frame)
            self._reused.pop(frame, None)
        else:
            self._keyframes.discard(frame)
            if "reused_from" in ev:
                self._reused[frame] = ev["reused_from"]
        self._refresh_markers()
        if frame == self._current_frame:
            self._show_masks_for(frame)  # 此时必命中缓存
        if self._exporting:
            self._export_on_frame(frame)
            return
        # 在途期间又有新 seek: 只补拉最新目标(仍是当前帧且未缓存才有意义)
        pending, self._gf_pending = self._gf_pending, None
        if (pending is not None and pending != frame
                and pending == self._current_frame
                and pending not in self._mask_cache
                and self._annot_ready()):
            self._request_frame_result(pending)

    # ================= 传播 =================
    def _on_submit_clicked(self) -> None:
        if not self._annot_ready() or self._journal or not self._prompts:
            return
        self._submit_pending = True
        self._worker.submit()  # 整段传播
        self._set_status("已提交, 等待传播开始…")
        self._refresh_ui_state()

    def _on_cancel_clicked(self) -> None:
        if self._worker is not None and self._propagating:
            self._worker.cancel()  # 线程安全, 直接调; cancelled 事件会出现在事件流里
            self._set_status("已请求取消…")

    def _on_propagate_event(self, ev: dict) -> None:
        etype = ev.get("type")
        if etype == "propagate_start":
            self._submit_pending = False
            self._propagating = True
            self._num_keyframes = int(ev.get("num_keyframes", 0))
            self._keyframe_count = 0
            self.timeline.set_progress(0.0)
            self.progress_label.setText(f"传播中: 0/{self._num_keyframes} 关键帧")
            self._set_status("传播开始…")
        elif etype == "prompts_applied":
            # 服务端已接收全部提示, 本地待提交计数清零
            self._dirty = 0
            self._refresh_dirty()
        elif etype == "keyframe":
            self._keyframe_count += 1
            frame = int(ev.get("frame_idx", -1))
            progress = ev.get("progress")
            if progress is not None:
                self.timeline.set_progress(float(progress))
            self._keyframes.add(frame)
            self._reused.pop(frame, None)
            masks = ev.get("mask_images")
            self._mask_cache[frame] = dict(masks) if masks else {}
            if frame == self._current_frame:
                self._show_masks_for(frame)
            self._refresh_markers()
            self.progress_label.setText(
                f"传播中: {self._keyframe_count}/{self._num_keyframes} 关键帧")
        elif etype in ("propagate_done", "cancelled"):
            self._propagating = False
            self._submit_pending = False
            self._submitted_once = True  # 取消也有部分结果, 可 seek 取帧
            self.timeline.set_progress(None)
            if etype == "propagate_done":
                self.progress_label.setText("传播完成")
                self._set_status("传播完成, 可逐帧查看或导出 Mask 序列")
            else:
                self.progress_label.setText("传播已取消")
                self._set_status("传播已取消(已算完的帧仍可取看)")
        # 无 type 的事件(如 cancel 的 ack)忽略
        self._refresh_ui_state()

    # ================= 重置追踪 =================
    def _on_reset_clicked(self) -> None:
        if not self._annot_ready() or self._journal:
            return
        ret = QMessageBox.question(
            self, "重置追踪",
            "将清空全部提示、掩码缓存与追踪状态(视频帧本身保留), 是否继续?")
        if ret != QMessageBox.Yes:
            return
        self._journal.append({"action": "reset", "undo": None})
        self._worker.reset()
        self._set_status("正在重置追踪…")
        self._refresh_ui_state()

    def _on_reset_done(self) -> None:
        # reset_video_tracking 语义: 服务端清空全部提示/物体/缓存/前沿, 保留帧入库状态。
        # 服务端清提示 -> 本地 _prompts 同步清空。
        self._mask_cache.clear()
        self._keyframes.clear()
        self._reused.clear()
        self._prompts.clear()
        self._dirty = 0
        self._submitted_once = False
        self._gf_pending = None
        self.canvas.clear_masks()
        self._refresh_dirty()
        self.progress_label.setText("")
        self._after_prompts_changed()
        self._set_status("追踪状态已重置(提示已清空, 视频帧保留)")

    # ================= prompt 文件 =================
    def _on_import_clicked(self) -> None:
        if not self._annot_ready() or self._pending_import is not None:
            return
        path, _ = QFileDialog.getOpenFileName(self, "导入提示文件", "",
                                              "Prompt 文件 (*.json)")
        if not path:
            return
        try:
            data = load_file(path)
        except ValueError as e:
            QMessageBox.warning(self, "导入失败", str(e))
            return
        if data.get("type") != "video":
            QMessageBox.warning(self, "导入失败",
                                "该文件是图像提示文件(type=image), 请选择视频提示文件")
            return
        mode = self._ask_merge_mode()
        if mode is None:
            return
        # 规范化: 服务端 load_video_prompt_file 对 points/labels 为 null 的组
        # 会抛 TypeError(len(None)), 这里统一转成 [] 并提前校验数量一致
        for fr in data.get("frames", []):
            for g in fr.get("groups") or []:
                g["points"] = g.get("points") or []
                g["labels"] = g.get("labels") or []
                if len(g["points"]) != len(g["labels"]):
                    QMessageBox.warning(
                        self, "导入失败",
                        f"帧 {fr.get('frame_idx')} 组 {g.get('group_id')}: "
                        f"点数({len(g['points'])})与标签数({len(g['labels'])})不一致")
                    return
        self._pending_import = (data, mode)
        self._journal.append({"action": "load_prompt_file", "undo": None})
        self._worker.load_prompt_file(data, mode)
        self._set_status(f"正在导入提示文件(合并方式: {mode})…")
        self._refresh_ui_state()

    def _ask_merge_mode(self) -> Optional[str]:
        """弹窗选择合并方式: append/replace/skip, 取消返回 None。"""
        box = QMessageBox(self)
        box.setWindowTitle("导入提示文件")
        box.setText("与已有提示冲突时的合并方式:")
        btn_append = box.addButton("追加(叠加到同帧同组)", QMessageBox.AcceptRole)
        btn_replace = box.addButton("覆盖(清空同帧同组再载入)", QMessageBox.ActionRole)
        btn_skip = box.addButton("跳过(保留已有同帧同组)", QMessageBox.ActionRole)
        box.addButton(QMessageBox.Cancel)
        box.exec()
        clicked = box.clickedButton()
        if clicked is btn_append:
            return "append"
        if clicked is btn_replace:
            return "replace"
        if clicked is btn_skip:
            return "skip"
        return None

    def _on_import_done(self, ev: dict) -> None:
        """command_result(load_prompt_file): 按服务端语义合并进本地 _prompts。"""
        pending, self._pending_import = self._pending_import, None
        if pending is None:
            return
        data, mode = pending
        loaded = 0
        for fr in data.get("frames", []):
            frame = fr["frame_idx"]
            for g in fr.get("groups") or []:
                gid = g["group_id"]
                pts = [(float(p[0]), float(p[1]), int(lbl))
                       for p, lbl in zip(g.get("points") or [],
                                         g.get("labels") or [])]
                box = (tuple(float(v) for v in g["box"])
                       if g.get("box") is not None else None)
                if not pts and box is None:
                    continue  # 全空组不载入(本地保持无空条目的不变式)
                frame_groups = self._prompts.setdefault(frame, {})
                exists = gid in frame_groups
                if mode == "skip" and exists:
                    continue
                if mode == "replace" and exists:
                    frame_groups[gid] = {"points": [], "box": None}
                entry = frame_groups.setdefault(gid, {"points": [], "box": None})
                entry["points"].extend(pts)          # append: 点叠加
                if box is not None:
                    entry["box"] = box               # append/replace: 框覆盖
                self.group_panel.ensure_group(gid)
                loaded += 1
        self._sync_group_colors()
        self._bump_dirty(loaded)
        self._after_prompts_changed()
        loaded_frames = ev.get("loaded_frames") or []
        if loaded_frames:
            # 服务端已使最早导入帧及之后的缓存失效, 本地镜像
            self._invalidate_cache_from(min(loaded_frames))
        self._set_status(f"提示文件已导入({mode}): 载入 {loaded} 组, "
                         f"涉及 {len(loaded_frames)} 帧")

    def _on_export_prompt_clicked(self) -> None:
        if not self._prompts:
            self._set_status("没有可导出的提示")
            return
        path, _ = QFileDialog.getSaveFileName(self, "导出提示文件",
                                              "video_prompt.json",
                                              "Prompt 文件 (*.json)")
        if not path:
            return
        try:
            save_video_file(path, self._build_export_frames())
        except OSError as e:
            QMessageBox.warning(self, "导出失败", str(e))
            return
        self._set_status(f"提示已导出: {path}")

    def _build_export_frames(self) -> List[dict]:
        """self._prompts -> save_video_file 的 frames 结构(空组跳过)。"""
        frames = []
        for frame in sorted(self._prompts):
            groups = []
            for gid in sorted(self._prompts[frame]):
                entry = self._prompts[frame][gid]
                if not entry["points"] and entry["box"] is None:
                    continue
                groups.append({
                    "group_id": gid,
                    "points": [[x, y] for x, y, _ in entry["points"]],
                    "labels": [lbl for _, _, lbl in entry["points"]],
                    "box": list(entry["box"]) if entry["box"] is not None else None,
                })
            if groups:
                frames.append({"frame_idx": frame, "groups": groups})
        return frames

    # ================= 导出 mask 序列 =================
    def _on_export_masks_clicked(self) -> None:
        if not (self._annot_ready() and self._submitted_once and not self._journal):
            return
        directory = QFileDialog.getExistingDirectory(self, "选择 Mask 导出目录")
        if not directory:
            return
        # 停播放, 避免导出链与播放 seek 争用 get_frame 通道
        self._play_timer.stop()
        self._playing = False
        self.timeline.set_playing(False)

        self._exporting = True
        self._export_stop = False
        self._export_state = {"dir": directory, "frame": 0, "saved": 0, "skipped": 0}
        dlg = QProgressDialog("正在导出 mask 序列…", "停止", 0, self._num_frames, self)
        dlg.setWindowTitle("导出 Mask 序列")
        dlg.setWindowModality(Qt.WindowModal)  # 模态挡住页面交互, 停不了导出链
        dlg.setMinimumDuration(0)
        dlg.canceled.connect(self._on_export_canceled)
        dlg.setValue(0)
        self._export_dialog = dlg
        self._set_status(f"正在导出 mask 序列到 {directory} …")
        self._refresh_ui_state()
        QTimer.singleShot(0, self._export_next)

    def _on_export_canceled(self) -> None:
        self._export_stop = True  # 标志位, 链式间隙检查

    def _export_next(self) -> None:
        """串行链式推进: 缓存命中直接存盘, 未命中发 get_frame 在结果里续链。"""
        st = self._export_state
        if st is None:
            return
        if self._export_stop or st["frame"] >= self._num_frames:
            self._finish_export()
            return
        frame = st["frame"]
        if self._export_dialog is not None:
            self._export_dialog.setValue(frame)
            self._export_dialog.setLabelText(f"正在导出第 {frame} 帧…")
        masks = self._mask_cache.get(frame)
        if masks is not None:
            self._export_save(frame, masks)
            st["frame"] += 1
            QTimer.singleShot(0, self._export_next)  # 逐帧让出事件循环, 停止按钮可点
        else:
            self._export_request(frame)

    def _export_request(self, frame: int) -> None:
        self._gf_inflight = True
        self._journal.append({"action": "get_frame", "undo": None, "meta": frame})
        self._worker.get_frame(frame, False)

    def _export_on_frame(self, frame: int) -> None:
        """get_frame 结果回到导出链(command_result 里推进下一帧)。"""
        st = self._export_state
        if st is None:
            return
        if frame != st["frame"]:
            self._export_request(st["frame"])  # 串行链不会错位, 防御性重发
            return
        self._export_save(frame, self._mask_cache.get(frame) or {})
        st["frame"] += 1
        QTimer.singleShot(0, self._export_next)

    def _export_save(self, frame: int, masks: Dict[int, Image.Image]) -> None:
        st = self._export_state
        if not masks:
            st["skipped"] += 1  # 无 mask 的帧跳过计数
            return
        for gid, img in masks.items():
            img.save(os.path.join(st["dir"], f"{frame:06d}_group{gid}.png"))
        st["saved"] += 1

    def _finish_export(self) -> None:
        st, self._export_state = self._export_state, None
        stopped = self._export_stop
        self._exporting = False
        self._export_stop = False
        if self._export_dialog is not None:
            self._export_dialog.reset()
            self._export_dialog.hide()
            self._export_dialog.deleteLater()
            self._export_dialog = None
        if st is not None:
            tail = "(已手动停止)" if stopped else ""
            self._set_status(f"Mask 导出结束{tail}: 保存 {st['saved']} 帧, "
                             f"跳过 {st['skipped']} 帧 -> {st['dir']}")
        self._refresh_ui_state()

    # ================= worker 事件 =================
    def _on_command_result(self, action: str, ev: dict) -> None:
        """命令确认: FIFO 出队; get_frame/reset/load_prompt_file 有后续处理。"""
        if self._journal:
            self._journal.popleft()
        if action == "get_frame":
            self._on_frame_result(ev)
        elif action == "reset":
            self._on_reset_done()
        elif action == "load_prompt_file":
            self._on_import_done(ev)
        # add_point/add_box/delete_point/clear_box/clear_group: 本地已乐观应用
        self._refresh_ui_state()

    # ================= 导出分割视频 =================
    def _on_export_video_clicked(self) -> None:
        if self._session_id is None or not self._mask_cache:
            self._set_status("尚无分割结果可导出, 请先提交传播")
            return
        default = f"segmented_{time.strftime('%Y%m%d_%H%M%S')}.mp4"
        path, _ = QFileDialog.getSaveFileName(
            self, "导出分割视频", default, "MP4 视频 (*.mp4)")
        if not path:
            return
        if self._playing:
            self._toggle_play()  # 停播放, 避免与导出线程争用解码器
        colors_bgr = {}
        for g in self.group_panel.visible_groups():
            c = self.group_panel.color_of(g)
            colors_bgr[g] = (c.blue(), c.green(), c.red())
        dlg = QProgressDialog("正在导出分割视频…", "停止", 0,
                              self._num_frames, self)
        dlg.setWindowTitle("导出分割视频")
        dlg.setWindowModality(Qt.WindowModal)
        dlg.setMinimumDuration(0)
        th = _VideoExportThread(self._video_path, path, self._fps,
                                self._num_frames, dict(self._mask_cache),
                                colors_bgr, parent=self)
        self._export_video_thread = th
        dlg.canceled.connect(th.cancel)
        th.progressed.connect(self._on_export_video_progress)
        th.done.connect(self._on_export_video_done)
        th.failed.connect(self._on_export_video_failed)
        th.finished.connect(th.deleteLater)
        self._export_dialog = dlg
        self._exporting = True
        self._refresh_ui_state()
        dlg.show()
        th.start()
        self._set_status(f"正在导出分割视频到 {path} …")

    def _on_export_video_progress(self, done: int, total: int) -> None:
        if self._export_dialog is not None:
            self._export_dialog.setMaximum(total)
            self._export_dialog.setValue(done)
            self._export_dialog.setLabelText(f"正在导出第 {done}/{total} 帧…")

    def _export_video_cleanup(self) -> None:
        if self._export_dialog is not None:
            self._export_dialog.hide()
            self._export_dialog.deleteLater()
            self._export_dialog = None
        self._export_video_thread = None
        self._exporting = False
        self._refresh_ui_state()

    def _on_export_video_done(self, complete: bool, path: str) -> None:
        self._export_video_cleanup()
        if complete:
            self._set_status(f"分割视频已导出: {path}")
        else:
            self._set_status(f"导出已停止, 保留部分文件: {path}")

    def _on_export_video_failed(self, msg: str) -> None:
        self._export_video_cleanup()
        QMessageBox.warning(self, "导出失败", msg)
        self._set_status(f"导出分割视频失败: {msg}")

    def _on_worker_failed(self, msg: str) -> None:
        """失败归口: 传播失败 / 命令失败(回滚) / 通用错误。

        worker 串行执行 + 信号 FIFO, 结合两个不变式可无歧义归类:
        - submit 只在 journal 为空时发出(_refresh_ui_state 保证),
          故 _submit_pending/_propagating 期间的 failed 必属 submit;
        - 其余时刻 failed 对应 journal 队首那条命令。
        """
        if self._propagating or self._submit_pending:
            self._propagating = False
            self._submit_pending = False
            self.timeline.set_progress(None)
            self.progress_label.setText("传播失败")
            self._set_status(f"传播失败: {msg}")
        elif self._journal:
            entry = self._journal.popleft()
            action = entry["action"]
            if action == "get_frame":
                self._gf_inflight = False
                self._gf_pending = None
                if self._exporting and self._export_state is not None:
                    # 导出链不因单帧失败中断: 记跳过并续链
                    self._export_state["skipped"] += 1
                    self._export_state["frame"] += 1
                    QTimer.singleShot(0, self._export_next)
                else:
                    self._set_status(f"取第 {entry.get('meta')} 帧结果失败: {msg}")
            elif action == "load_prompt_file":
                self._pending_import = None
                self._set_status(f"导入提示文件失败: {msg}")
            elif action == "reset":
                self._set_status(f"重置追踪失败: {msg}")
            else:
                undo: Optional[Callable[[], None]] = entry.get("undo")
                if undo is not None:
                    undo()
                self._bump_dirty(-1)
                self._after_prompts_changed()
                self._set_status(f"操作失败, 已回滚本地提示: {msg}")
        else:
            self._set_status(f"后台错误: {msg}")
        self._refresh_ui_state()

    def _on_worker_closed(self) -> None:
        """WS 关闭: 主动关闭忽略; 意外断开则禁用交互并提示重开会话。"""
        if self._worker is None or self._expected_close:
            return
        self._broken = True
        self._propagating = False
        self._submit_pending = False
        self._journal.clear()  # 通道已死, 未确认命令无法配对/回滚
        self._gf_inflight = False
        self._gf_pending = None
        self.timeline.set_progress(None)
        self._set_status("连接已断开: 请[关闭会话]后重新上传视频")
        self._refresh_ui_state()

    # ================= 退出 =================
    def shutdown(self) -> None:
        """页面清理(主窗口 closeEvent 调): 停导出/播放, 关 worker 与会话。"""
        self._export_stop = True
        if self._export_video_thread is not None:
            self._export_video_thread.cancel()  # 协程式取消, 不阻塞关闭
        if self._export_dialog is not None:
            self._export_dialog.hide()
            self._export_dialog.deleteLater()
            self._export_dialog = None
        self._play_timer.stop()
        self._close_session()

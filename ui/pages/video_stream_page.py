"""视频实时流页: 多源实时采集 + 流式推帧跟踪 + 定格标注(仅最新帧可标)。

工作流程:
1. 顶部选择源(摄像头/视频文件/网络流 URL/模拟流)并开始推流:
   创建流式会话(auto_predict=True) -> VideoSessionWorker(WS 驱动) + CaptureThread(采集)
2. 逐帧: 采集线程发 RGB numpy -> 画布显示 + 推帧;
   在途未回结果超过 2 帧时跳过当次推帧(显示照常), 页面侧限流保证 worker 不再丢帧,
   从而使本地推帧计数 _push_idx 与服务端 received_frame_count-1 始终一致(标注要用)
3. 服务端逐帧回跟踪掩码 -> 仅当结果属于最新推帧(frame_idx == _push_idx)时上屏
4. "暂停并标注": 采集暂停, 画面定格在最新帧; 标注命令一律带 frame_idx=_push_idx
   (服务端校验只能给最新收到的帧加提示); auto_predict 即时重算, 结果带掩码直接刷新
5. "继续推流": 清掉属于旧帧的画布提示与掩码, 恢复采集
"""

from __future__ import annotations

import os
import time
from collections import deque
from typing import Deque, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image
from PySide6.QtCore import Qt, QThread, QTimer, Signal
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (QApplication, QButtonGroup, QCheckBox, QComboBox,
                               QFileDialog, QHBoxLayout, QLabel, QLineEdit,
                               QMessageBox, QPushButton, QRadioButton,
                               QSlider, QSpinBox, QStackedWidget, QVBoxLayout,
                               QWidget)

from ..context import AppContext
from ..utils import pil_to_qimage
from ..widgets.canvas import AnnotationCanvas
from ..widgets.group_panel import GroupPanel
from ..workers import VideoSessionWorker, run_api

# 源类型(下拉顺序即索引)
_SRC_CAMERA = 0
_SRC_FILE = 1
_SRC_URL = 2
_SRC_MOCK = 3
_SRC_KIND = {_SRC_CAMERA: "camera", _SRC_FILE: "file",
             _SRC_URL: "url", _SRC_MOCK: "mock"}

# 在途(已推未回结果)帧上限: 超过则跳过一次推帧, 防止 worker 侧丢帧导致帧号漂移
_MAX_INFLIGHT = 2
# FPS 滑窗(秒)
_FPS_WINDOW = 3.0

# 本地提示记账类型
_Point = Tuple[float, float, int, int]           # (x, y, label, gid)
_Box = Tuple[float, float, float, float, int]    # (x1, y1, x2, y2, gid)


class CaptureThread(QThread):
    """采集线程: 摄像头/视频文件/网络流/模拟噪声流 -> frame_ready(RGB numpy)。

    视频文件按源 fps 节流并循环播放; 摄像头/网络流靠 read() 阻塞取自然帧率
    (若驱动上报了有效 fps 也做节流, 防止个别源狂转); 模拟流 640x480 暖色噪声 30fps。
    set_paused(True) 后不再发帧; 实时源(摄像头/URL)暂停期间持续读并丢弃,
    避免恢复时读到积压的旧帧。
    """

    frame_ready = Signal(object)  # RGB uint8 numpy (H, W, 3)
    error = Signal(str)

    def __init__(self, kind: str, param, parent=None):
        super().__init__(parent)
        self._kind = kind            # "camera" / "file" / "url" / "mock"
        self._param = param          # camera=int 索引; file/url=str; mock=None
        self._running = False
        self._paused = False

    # ---------------- 控制(主线程调用) ----------------
    def set_paused(self, on: bool) -> None:
        self._paused = bool(on)

    def stop(self) -> None:
        """置停止标志并等待线程结束(干净释放视频源)。"""
        self._running = False
        self.wait(3000)

    # ---------------- 线程体 ----------------
    def run(self) -> None:
        self._running = True
        if self._kind == "mock":
            self._run_mock()
        else:
            self._run_capture()

    def _run_mock(self) -> None:
        interval = 1.0 / 30.0
        while self._running:
            if self._paused:
                time.sleep(0.05)
                continue
            t0 = time.perf_counter()
            # 暖色噪声: 提升 R 通道
            arr = np.random.randint(0, 200, (480, 640, 3), dtype=np.uint8)
            arr[..., 0] = np.clip(arr[..., 0].astype(np.int16) + 60,
                                  0, 255).astype(np.uint8)
            self.frame_ready.emit(arr)
            self._throttle(t0, interval)

    def _run_capture(self) -> None:
        cap = cv2.VideoCapture(self._param)
        if not cap.isOpened():
            cap.release()
            self.error.emit(f"无法打开视频源: {self._param}")
            return
        is_file = self._kind == "file"
        is_live = self._kind in ("camera", "url")
        fps = cap.get(cv2.CAP_PROP_FPS)
        default_fps = 25.0 if is_file else 30.0
        interval = 1.0 / fps if fps and fps > 1.0 else 1.0 / default_fps
        fails = 0  # 连续读帧失败计数(防死循环)
        try:
            while self._running:
                if self._paused:
                    if is_live:
                        cap.read()          # 丢弃缓冲帧, 恢复时拿到最新画面
                        time.sleep(0.005)
                    else:
                        time.sleep(0.05)    # 文件/mock 暂停即冻结, 不前进
                    continue
                t0 = time.perf_counter()
                ret, frame = cap.read()
                if not ret:
                    fails += 1
                    if is_file and fails <= 5:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # 循环播放
                        continue
                    self.error.emit(f"视频源中断: {self._param}")
                    break
                fails = 0
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.frame_ready.emit(np.ascontiguousarray(rgb))
                self._throttle(t0, interval)
        finally:
            cap.release()

    @staticmethod
    def _throttle(t0: float, interval: float) -> None:
        """扣除本帧处理耗时后再 sleep, 避免累积延迟。"""
        dt = time.perf_counter() - t0
        if interval > dt:
            time.sleep(interval - dt)


class VideoStreamPage(QWidget):
    """视频实时流页: 源控制 + 实时推流跟踪 + 定格标注。"""

    def __init__(self, context: AppContext, parent=None):
        super().__init__(parent)
        self._ctx = context

        # ---- 会话状态 ----
        self._session_id: Optional[str] = None
        self._worker: Optional[VideoSessionWorker] = None
        self._capture: Optional[CaptureThread] = None
        self._pending_source: Tuple[str, object] = ("mock", None)  # 建会话成功后用
        self._creating = False        # 会话创建中(HTTP 在途)
        self._session_active = False  # 会话 + 采集 + worker 均在跑
        self._annotating = False      # 标注态(采集暂停, 画面定格)
        self._closing = False         # shutdown 后忽略一切异步回调

        # ---- 帧计数 / 在途控制 ----
        self._push_idx = -1           # 最近一次推帧的帧号(= 服务端 received-1, 第 1 帧为 0)
        self._frame_seq = 0           # 采集帧计数(仅统计)
        self._push_times: Deque[Tuple[int, float]] = deque()  # 未回结果的推帧(FIFO)

        # ---- 统计 ----
        self._push_win: Deque[float] = deque()    # 推帧时刻滑窗
        self._result_win: Deque[float] = deque()  # 回结果时刻滑窗
        self._last_latency_ms: Optional[float] = None

        # ---- 当前帧提示本地记账(仅标注态, 用于画布渲染) ----
        self._prompt_points: List[_Point] = []
        self._prompt_boxes: List[_Box] = []
        self._suppress_group_removed = False  # 程序化清组时屏蔽 group_removed 回环

        self._setup_ui()
        self._setup_shortcuts()
        self._status_timer = QTimer(self)
        self._status_timer.setInterval(500)
        self._status_timer.timeout.connect(self._refresh_status)
        self._status_timer.start()
        self._refresh_controls()
        self._refresh_status()

    # ================= UI =================
    def _setup_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        # ---- 顶部源控制条 ----
        src_bar = QHBoxLayout()
        src_bar.addWidget(QLabel("源类型:"))
        self.src_combo = QComboBox()
        self.src_combo.addItems(["摄像头", "视频文件", "网络流 URL", "模拟流"])
        self.src_combo.currentIndexChanged.connect(self._on_src_changed)
        src_bar.addWidget(self.src_combo)

        self.src_stack = QStackedWidget()
        # 摄像头: 索引
        cam_page = QWidget()
        h = QHBoxLayout(cam_page)
        h.setContentsMargins(0, 0, 0, 0)
        h.addWidget(QLabel("索引:"))
        self.cam_spin = QSpinBox()
        self.cam_spin.setRange(0, 16)
        h.addWidget(self.cam_spin)
        h.addStretch()
        self.src_stack.addWidget(cam_page)
        # 视频文件: 路径 + 浏览
        file_page = QWidget()
        h = QHBoxLayout(file_page)
        h.setContentsMargins(0, 0, 0, 0)
        self.file_edit = QLineEdit()
        self.file_edit.setPlaceholderText("视频文件路径")
        browse_btn = QPushButton("浏览...")
        browse_btn.setFocusPolicy(Qt.NoFocus)
        browse_btn.clicked.connect(self._on_browse_file)
        h.addWidget(self.file_edit, stretch=1)
        h.addWidget(browse_btn)
        self.src_stack.addWidget(file_page)
        # 网络流: URL
        url_page = QWidget()
        h = QHBoxLayout(url_page)
        h.setContentsMargins(0, 0, 0, 0)
        self.url_edit = QLineEdit()
        self.url_edit.setPlaceholderText("rtsp://... 或 http://...")
        h.addWidget(self.url_edit)
        self.src_stack.addWidget(url_page)
        # 模拟流: 无参数
        mock_page = QWidget()
        h = QHBoxLayout(mock_page)
        h.setContentsMargins(0, 0, 0, 0)
        h.addWidget(QLabel("内置暖色噪声流 640×480 @ 30fps"))
        h.addStretch()
        self.src_stack.addWidget(mock_page)
        src_bar.addWidget(self.src_stack, stretch=1)

        src_bar.addWidget(QLabel("关键帧间隔:"))
        self.stride_spin = QSpinBox()
        self.stride_spin.setRange(1, 60)
        self.stride_spin.setValue(1)
        self.stride_spin.setToolTip("frame_stride: 每 N 个推帧做一次关键帧计算")
        src_bar.addWidget(self.stride_spin)

        self.start_btn = QPushButton("开始推流")
        self.stop_btn = QPushButton("停止")
        for b in (self.start_btn, self.stop_btn):
            b.setFocusPolicy(Qt.NoFocus)
        self.start_btn.clicked.connect(self._on_start)
        self.stop_btn.clicked.connect(self._on_stop)
        src_bar.addWidget(self.start_btn)
        src_bar.addWidget(self.stop_btn)
        root.addLayout(src_bar)

        # ---- 中部: 组面板 + 画布 ----
        body = QHBoxLayout()
        self.group_panel = GroupPanel()
        self.group_panel.setFixedWidth(220)
        self.canvas = AnnotationCanvas()
        body.addWidget(self.group_panel)
        body.addWidget(self.canvas, stretch=1)
        root.addLayout(body, stretch=1)

        # ---- 标注控制条 ----
        ctrl = QHBoxLayout()
        self.annotate_btn = QPushButton("暂停并标注")
        self.annotate_btn.setFocusPolicy(Qt.NoFocus)
        self.annotate_btn.clicked.connect(self._on_toggle_annotate)
        ctrl.addWidget(self.annotate_btn)

        ctrl.addWidget(QLabel(" 模式:"))
        self.point_radio = QRadioButton("点(P)")
        self.box_radio = QRadioButton("框(B)")
        self.point_radio.setChecked(True)
        self.point_radio.toggled.connect(
            lambda on: on and self._set_mode("point"))
        self.box_radio.toggled.connect(
            lambda on: on and self._set_mode("box"))
        ctrl.addWidget(self.point_radio)
        ctrl.addWidget(self.box_radio)

        self.undo_point_btn = QPushButton("撤销点")
        self.clear_box_btn = QPushButton("清框")
        self.clear_group_btn = QPushButton("清组")
        self.reset_btn = QPushButton("重置追踪")
        for b in (self.undo_point_btn, self.clear_box_btn,
                  self.clear_group_btn, self.reset_btn):
            b.setFocusPolicy(Qt.NoFocus)
        self.undo_point_btn.clicked.connect(self._on_undo_point)
        self.clear_box_btn.clicked.connect(self._on_clear_box)
        self.clear_group_btn.clicked.connect(self._on_clear_group)
        self.reset_btn.clicked.connect(self._on_reset_tracking)
        ctrl.addWidget(self.undo_point_btn)
        ctrl.addWidget(self.clear_box_btn)
        ctrl.addWidget(self.clear_group_btn)
        ctrl.addWidget(self.reset_btn)

        ctrl.addWidget(QLabel(" 正负:"))
        self.pos_radio = QRadioButton("正点")
        self.neg_radio = QRadioButton("负点")
        self.pos_radio.setChecked(True)
        # 同父 widget 的 QRadioButton 默认自动互斥, 模式组与正负组必须显式分组隔离
        mode_group = QButtonGroup(self)
        mode_group.addButton(self.point_radio)
        mode_group.addButton(self.box_radio)
        label_group = QButtonGroup(self)
        label_group.addButton(self.pos_radio)
        label_group.addButton(self.neg_radio)
        self.pos_radio.toggled.connect(self._on_label_mode_changed)
        self.lr_check = QCheckBox("左右键定正负")
        self.lr_check.setChecked(True)
        self.lr_check.setToolTip("勾选: 左键正点/右键负点; 不勾: 点击用当前单选的正负")
        self.lr_check.toggled.connect(self._on_label_mode_changed)
        ctrl.addWidget(self.pos_radio)
        ctrl.addWidget(self.neg_radio)
        ctrl.addWidget(self.lr_check)

        ctrl.addWidget(QLabel(" 透明度:"))
        self.opacity_slider = QSlider(Qt.Horizontal)
        self.opacity_slider.setRange(0, 255)
        self.opacity_slider.setValue(128)
        self.opacity_slider.setFixedWidth(120)
        self.opacity_slider.valueChanged.connect(self.canvas.set_opacity)
        ctrl.addWidget(self.opacity_slider)
        ctrl.addStretch()
        root.addLayout(ctrl)

        # ---- 底部状态行 ----
        stat = QHBoxLayout()
        self.session_label = QLabel("未开始")
        self.push_fps_label = QLabel("推帧: --")
        self.result_fps_label = QLabel("回结果: --")
        self.latency_label = QLabel("延迟: --")
        self.frame_label = QLabel("帧号: --")
        labels = (self.session_label, self.push_fps_label,
                  self.result_fps_label, self.latency_label, self.frame_label)
        for i, lbl in enumerate(labels):
            if i:
                stat.addWidget(QLabel(" | "))
            stat.addWidget(lbl)
        stat.addStretch()
        root.addLayout(stat)

        # ---- 面板 / 画布信号 ----
        self.group_panel.group_removed.connect(self._on_group_removed)
        self.group_panel.visibility_changed.connect(self._on_visibility_changed)
        self.canvas.point_added.connect(self._on_point_added)
        self.canvas.box_drawn.connect(self._on_box_drawn)
        self._on_label_mode_changed()  # 初始化画布 label 模式

    def _setup_shortcuts(self) -> None:
        """空格=暂停/继续, P=点模式, B=框模式(仅会话期间启用, 避免吃输入框按键)。"""
        self._sc_space = QShortcut(QKeySequence(Qt.Key_Space), self)
        self._sc_space.activated.connect(self._on_space_shortcut)
        self._sc_p = QShortcut(QKeySequence("P"), self)
        self._sc_p.activated.connect(lambda: self._on_mode_shortcut("point"))
        self._sc_b = QShortcut(QKeySequence("B"), self)
        self._sc_b.activated.connect(lambda: self._on_mode_shortcut("box"))
        for sc in (self._sc_space, self._sc_p, self._sc_b):
            sc.setEnabled(False)

    def _on_space_shortcut(self) -> None:
        """空格快捷键: 输入框聚焦(如组重命名)时还原为普通输入。"""
        fw = QApplication.focusWidget()
        if isinstance(fw, QLineEdit):
            fw.insert(" ")
            return
        self._on_toggle_annotate()

    def _on_mode_shortcut(self, mode: str) -> None:
        """P/B 快捷键: 输入框聚焦时还原为普通输入。"""
        fw = QApplication.focusWidget()
        if isinstance(fw, QLineEdit):
            fw.insert(mode[0])
            return
        self._set_mode(mode)

    # ================= 源控制 =================
    def _on_src_changed(self, idx: int) -> None:
        self.src_stack.setCurrentIndex(idx)

    def _on_browse_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "选择视频文件", "",
            "视频文件 (*.mp4 *.avi *.mov *.mkv *.flv *.wmv);;所有文件 (*)")
        if path:
            self.file_edit.setText(path)

    def _current_source(self) -> Optional[Tuple[str, object]]:
        """读取并校验当前源参数, 不合法时弹窗并返回 None。"""
        idx = self.src_combo.currentIndex()
        kind = _SRC_KIND[idx]
        if kind == "camera":
            return kind, self.cam_spin.value()
        if kind == "file":
            path = self.file_edit.text().strip()
            if not path or not os.path.isfile(path):
                QMessageBox.warning(self, "参数错误", "请选择存在的视频文件")
                return None
            return kind, path
        if kind == "url":
            url = self.url_edit.text().strip()
            if not url:
                QMessageBox.warning(self, "参数错误", "请输入网络流 URL")
                return None
            return kind, url
        return kind, None  # mock 无参数

    # ================= 开始 / 停止 =================
    def _on_start(self) -> None:
        if self._session_active or self._creating:
            return
        source = self._current_source()
        if source is None:
            return
        self._pending_source = source
        self._creating = True
        self._refresh_controls()
        self._refresh_status()
        run_api(self._ctx.client.create_stream_session,
                self.stride_spin.value(), True,
                on_ok=self._on_session_created,
                on_err=self._on_session_create_failed,
                parent=self)

    def _on_session_created(self, sid) -> None:
        self._creating = False
        if self._closing:
            # 页面已关闭: 归还刚创建的会话
            run_api(self._ctx.client.close_video_session, str(sid))
            return
        self._session_id = str(sid)
        self._reset_counters()

        self._worker = VideoSessionWorker(self._ctx.client, self._session_id,
                                          parent=self)
        self._worker.frame_result.connect(self._on_frame_result)
        self._worker.command_result.connect(self._on_command_result)
        self._worker.failed.connect(self._on_worker_failed)
        self._worker.closed.connect(self._on_worker_closed)
        self._worker.closed.connect(self._worker.deleteLater)
        self._worker.start()

        kind, param = self._pending_source
        self._capture = CaptureThread(kind, param, parent=self)
        self._capture.frame_ready.connect(self._on_frame)
        self._capture.error.connect(self._on_capture_error)
        self._capture.start()

        self._session_active = True
        self._refresh_controls()
        self._refresh_status()
        self._ctx.status_message.emit(f"视频流会话已建立: {self._session_id}")

    def _on_session_create_failed(self, msg: str) -> None:
        self._creating = False
        self._refresh_controls()
        self._refresh_status()
        if not self._closing:
            QMessageBox.warning(self, "创建会话失败", f"无法创建视频流会话:\n{msg}")
        self._ctx.status_message.emit(f"创建视频流会话失败: {msg}")

    def _on_stop(self) -> None:
        self._stop_session()
        self._ctx.status_message.emit("视频流已停止")

    def _stop_session(self) -> None:
        """停采集 -> 关闭 worker -> 归还会话(可重入, 幂等)。"""
        if self._capture is not None:
            cap = self._capture
            self._capture = None
            cap.stop()  # flag + wait, 干净退出
            self._defer_delete_thread(cap)
        if self._worker is not None:
            worker = self._worker
            self._worker = None
            # 预期内关闭: 断开页面回调, 仅保留 closed->deleteLater
            for sig, slot in ((worker.frame_result, self._on_frame_result),
                              (worker.command_result, self._on_command_result),
                              (worker.failed, self._on_worker_failed),
                              (worker.closed, self._on_worker_closed)):
                try:
                    sig.disconnect(slot)
                except (RuntimeError, TypeError):
                    pass
            worker.shutdown()
        if self._session_id is not None:
            # fire-and-forget 归还会话
            run_api(self._ctx.client.close_video_session, self._session_id)
            self._session_id = None
        self._session_active = False
        self._annotating = False
        self._reset_counters()
        self._refresh_controls()
        self._refresh_status()

    @staticmethod
    def _defer_delete_thread(thread: QThread) -> None:
        """安全销毁 QThread: 已退出则 deleteLater; 仍在跑(源阻塞)则等 finished 再删,
        避免销毁仍在运行的线程导致崩溃。"""
        if thread.isRunning():
            thread.finished.connect(thread.deleteLater)
        else:
            thread.deleteLater()

    def _reset_counters(self) -> None:
        """帧号/FPS/提示/画布/组面板全部复位(开始新会话或停止时)。"""
        self._push_idx = -1
        self._frame_seq = 0
        self._push_times.clear()
        self._push_win.clear()
        self._result_win.clear()
        self._last_latency_ms = None
        self._annotating = False
        self.annotate_btn.setText("暂停并标注")
        self._clear_annotation_state()
        self._suppress_group_removed = True
        self.group_panel.clear()
        self._suppress_group_removed = False

    # ================= 逐帧处理 =================
    def _on_frame(self, arr: np.ndarray) -> None:
        """采集帧回调(主线程): 显示 + 限流推帧。"""
        if not self._session_active or self._annotating or self._worker is None:
            return  # 标注态/已停止: 丢弃迟到帧, 画面定格
        self._frame_seq += 1
        pil = Image.fromarray(arr, "RGB")
        self.canvas.set_image(pil_to_qimage(pil))  # 同尺寸换帧不重置视图
        now = time.perf_counter()
        # 服务端推帧处理出错时不回结果(error 事件走 failed), 在途条目会永久占住
        # 限流名额导致推流停摆; 超过 5 秒未回的视为丢失, 丢弃以恢复推流
        while self._push_times and now - self._push_times[0][1] > 5.0:
            lost, _ = self._push_times.popleft()
            self._ctx.status_message.emit(
                f"警告: 第 {lost} 帧结果超时未回, 已丢弃其在途记录")
        # 在途控制: 积压超过上限则跳过一次推帧(显示照常)
        if len(self._push_times) < _MAX_INFLIGHT:
            self._push_idx += 1
            self._push_times.append((self._push_idx, now))
            self._push_win.append(now)
            self._worker.push_frame(pil)

    def _on_frame_result(self, res: dict) -> None:
        """推帧结果: 统计 + 仅最新帧掩码上屏。"""
        now = time.perf_counter()
        self._result_win.append(now)
        if self._push_times:
            idx, t0 = self._push_times.popleft()  # 结果按序返回, 配队首
            ridx = res.get("frame_idx", idx)
            if ridx != idx:
                self._ctx.status_message.emit(
                    f"警告: 帧序号漂移(本地 {idx} / 服务端 {ridx})")
            self._last_latency_ms = (now - t0) * 1000.0
        # 结果按序到达, 收到的就是最新可用掩码; 分割慢时画面会领先结果几帧,
        # 无条件上屏(用稍旧的掩码顶替), 避免"只显示最新帧结果"导致掩码冻结
        self._apply_masks(res)
        self._refresh_status()

    def _apply_masks(self, res: dict) -> None:
        """事件里的 mask_images({gid: PIL}) -> 画布; 无掩码且空组则清空。"""
        masks = res.get("mask_images")
        if masks:
            self.canvas.set_masks({int(g): pil_to_qimage(m)
                                   for g, m in masks.items()})
        elif not res.get("groups"):
            self.canvas.clear_masks()

    # ================= 标注 =================
    def _on_toggle_annotate(self) -> None:
        if self._closing:
            return
        if not self._session_active or self._worker is None:
            self._ctx.status_message.emit("请先开始推流")
            return
        if not self._annotating:
            if self._push_idx < 0:
                self._ctx.status_message.emit("等待第一帧到达后再标注")
                return
            self._annotating = True
            if self._capture is not None:
                self._capture.set_paused(True)
            self.annotate_btn.setText("继续推流")
            self._ctx.status_message.emit(
                "已定格最新帧, 可标注(仅当前帧可标)")
        else:
            self._annotating = False
            # 旧帧的提示与掩码对后续帧无效, 恢复前清掉
            self._clear_annotation_state()
            if self._capture is not None:
                self._capture.set_paused(False)
            self.annotate_btn.setText("暂停并标注")
            self._ctx.status_message.emit("继续推流")
        self._refresh_controls()
        self._refresh_status()

    def _annotation_gate(self) -> bool:
        """画布点击准入: 非标注态不生效并提示。"""
        if self._annotating and self._worker is not None:
            return True
        if self._session_active:
            self._ctx.status_message.emit(
                "推流中: 可直接单击加点, 画框/撤销需先暂停并标注")
        else:
            self._ctx.status_message.emit("请先开始推流")
        return False

    def _live_ready(self) -> bool:
        """实时加点准入: 会话在推、非定格标注态、已有推帧。"""
        return (self._session_active and self._worker is not None
                and not self._annotating and self._push_idx >= 0)

    def _current_gid(self) -> int:
        """取当前组, 无组时自动新建。"""
        gid = self.group_panel.current_group()
        if gid is None:
            gid = self.group_panel.add_group()
        return gid

    def _on_point_added(self, x: float, y: float, label: int) -> None:
        if self._live_ready():
            # 实时加点: 坐标绑定当前最新推帧; 命令与推帧同队 FIFO,
            # 执行时之前的推帧已全部到达服务端, 帧号天然对齐"最新帧"约束
            gid = self._current_gid()
            self._worker.add_point(gid, x, y, label, self._push_idx)
            self._ctx.status_message.emit(
                f"实时加点: 组{gid} {'正' if label else '负'}点 -> "
                f"第 {self._push_idx} 帧(掩码随后续推帧出现)")
            return
        if not self._annotation_gate():
            return
        gid = self._current_gid()
        self._prompt_points.append((x, y, label, gid))
        self.canvas.set_prompts(self._prompt_points, self._prompt_boxes)
        self._worker.add_point(gid, x, y, label, self._push_idx)

    def _on_box_drawn(self, x1: float, y1: float,
                      x2: float, y2: float) -> None:
        if not self._annotation_gate():
            return
        gid = self._current_gid()
        # 服务端 set_box 语义: 每组只保留一个框
        self._prompt_boxes = [b for b in self._prompt_boxes if b[4] != gid]
        self._prompt_boxes.append((x1, y1, x2, y2, gid))
        self.canvas.set_prompts(self._prompt_points, self._prompt_boxes)
        self._worker.add_box(gid, x1, y1, x2, y2, self._push_idx)

    def _on_undo_point(self) -> None:
        if not self._annotation_gate():
            return
        gid = self.group_panel.current_group()
        if gid is None:
            self._ctx.status_message.emit("请先选择组")
            return
        for i in range(len(self._prompt_points) - 1, -1, -1):
            if self._prompt_points[i][3] == gid:
                del self._prompt_points[i]
                break
        else:
            self._ctx.status_message.emit("当前组没有可撤销的点")
            return
        self.canvas.set_prompts(self._prompt_points, self._prompt_boxes)
        self._worker.delete_point(gid, self._push_idx, -1)

    def _on_clear_box(self) -> None:
        if not self._annotation_gate():
            return
        gid = self.group_panel.current_group()
        if gid is None:
            self._ctx.status_message.emit("请先选择组")
            return
        new_boxes = [b for b in self._prompt_boxes if b[4] != gid]
        if len(new_boxes) == len(self._prompt_boxes):
            self._ctx.status_message.emit("当前组没有框")
            return
        self._prompt_boxes = new_boxes
        self.canvas.set_prompts(self._prompt_points, self._prompt_boxes)
        self._worker.clear_box(gid, self._push_idx)

    def _on_clear_group(self) -> None:
        if not self._annotation_gate():
            return
        gid = self.group_panel.current_group()
        if gid is None:
            self._ctx.status_message.emit("请先选择组")
            return
        self._worker.clear_group(gid)  # 面板/画布清理由 command_result 完成

    def _on_reset_tracking(self) -> None:
        if not self._session_active or self._worker is None:
            self._ctx.status_message.emit("请先开始推流")
            return
        ret = QMessageBox.question(
            self, "重置追踪", "确定清空全部提示并重置追踪状态吗?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if ret == QMessageBox.Yes:
            self._worker.reset()  # 清渲染在 command_result 里做

    def _on_command_result(self, action: str, res: dict) -> None:
        """标注命令结果: auto_predict 重算后带掩码的直接刷新。"""
        if self._closing:
            return
        # get_frame 无掩码(空组)时也要走 _apply_masks 清掉陈旧掩码;
        # 掩码无条件上屏: 结果按序到达, 收到的即最新可用(画面可能领先几帧)
        if (res.get("mask_images") is not None
                or action in ("add_point", "add_box", "get_frame")):
            self._apply_masks(res)
        if action == "clear_group":
            gid = res.get("group_id")
            if gid is not None:
                self._purge_group(int(gid))
        elif action in ("delete_point", "clear_box"):
            # 删除类命令不回掩码(auto_predict 也不重算), 主动取帧刷新
            self._refresh_after_delete()
        elif action == "reset":
            self._clear_annotation_state()
        self._refresh_status()

    def _refresh_after_delete(self) -> None:
        """删除提示后服务端不重算, 用 get_frame 拿当前帧最新结果。"""
        if self._worker is None or not self._session_active:
            return
        if self._prompt_points or self._prompt_boxes:
            self._worker.get_frame(self._push_idx, compute_if_missing=True)
        else:
            self.canvas.clear_masks()

    def _purge_group(self, gid: int) -> None:
        """清除某组的本地痕迹(提示/掩码/面板行)。"""
        self._prompt_points = [p for p in self._prompt_points if p[3] != gid]
        self._prompt_boxes = [b for b in self._prompt_boxes if b[4] != gid]
        self.canvas.set_prompts(self._prompt_points, self._prompt_boxes)
        self.canvas.remove_mask(gid)
        self._suppress_group_removed = True
        self.group_panel.remove_group(gid)
        self._suppress_group_removed = False

    def _clear_annotation_state(self) -> None:
        """清画布提示与掩码(属于旧帧), 保留组面板。"""
        self._prompt_points = []
        self._prompt_boxes = []
        self.canvas.set_prompts([], [])
        self.canvas.clear_masks()

    # ================= 组面板 =================
    def _on_group_removed(self, gid: int) -> None:
        if self._suppress_group_removed:
            return
        self.canvas.remove_mask(gid)
        self._prompt_points = [p for p in self._prompt_points if p[3] != gid]
        self._prompt_boxes = [b for b in self._prompt_boxes if b[4] != gid]
        self.canvas.set_prompts(self._prompt_points, self._prompt_boxes)
        if self._session_active and self._worker is not None:
            self._worker.clear_group(gid)  # 服务端同步删除(command_result 幂等清理)

    def _on_visibility_changed(self, gid: int, visible: bool) -> None:
        hidden = {g for g in self.group_panel.groups()
                  if g not in self.group_panel.visible_groups()}
        self.canvas.set_hidden_groups(hidden)

    # ================= 标注模式 / 正负 =================
    def _set_mode(self, mode: str) -> None:
        if mode == "point":
            self.point_radio.setChecked(True)
        else:
            self.box_radio.setChecked(True)
        self.canvas.set_mode(mode)

    def _on_label_mode_changed(self) -> None:
        lr = self.lr_check.isChecked()
        self.pos_radio.setEnabled(not lr)
        self.neg_radio.setEnabled(not lr)
        self.canvas.set_button_label_mode(lr)
        self.canvas.set_label(1 if self.pos_radio.isChecked() else 0)

    # ================= 异常 / 状态 =================
    def _on_worker_failed(self, msg: str) -> None:
        if self._closing:
            return
        if "没有已计算的关键帧" in msg:
            # 删光提示后 get_frame 的良性落空: 无结果可显示
            self.canvas.clear_masks()
            return
        self.session_label.setText(f"错误: {msg}")
        self._ctx.status_message.emit(f"视频流错误: {msg}")

    def _on_worker_closed(self) -> None:
        """非预期关闭(正常停止已在 _stop_session 断开此回调): 复位到未开始。"""
        if self._closing:
            return
        self._worker = None  # deleteLater 已在建 worker 时连接
        if self._capture is not None:
            cap = self._capture
            self._capture = None
            cap.stop()
            self._defer_delete_thread(cap)
        if self._session_id is not None:
            run_api(self._ctx.client.close_video_session, self._session_id)
            self._session_id = None
        self._session_active = False
        self._annotating = False
        self._reset_counters()
        self._refresh_controls()
        self._refresh_status()
        self.session_label.setText("会话意外中断")
        self._ctx.status_message.emit("视频会话通道已断开")

    def _on_capture_error(self, msg: str) -> None:
        if self._closing:
            return
        self._ctx.status_message.emit(f"采集错误: {msg}")
        self._stop_session()
        self.session_label.setText(f"采集错误: {msg}")

    # ================= 状态行 / 控件使能 =================
    @staticmethod
    def _fps(win: Deque[float], now: float) -> float:
        """滑窗 FPS。"""
        while win and now - win[0] > _FPS_WINDOW:
            win.popleft()
        if len(win) < 2:
            return 0.0
        span = win[-1] - win[0]
        return (len(win) - 1) / span if span > 0 else 0.0

    def _refresh_status(self) -> None:
        now = time.perf_counter()
        if self._creating:
            self.session_label.setText("正在创建会话...")
        elif self._session_active:
            self.session_label.setText(
                "标注中(仅当前帧)" if self._annotating else "推流中")
        elif not self.session_label.text().startswith(("错误", "会话意外", "采集错误")):
            self.session_label.setText("未开始")
        if not self._session_active:
            self.push_fps_label.setText("推帧: --")
            self.result_fps_label.setText("回结果: --")
            self.latency_label.setText("延迟: --")
            self.frame_label.setText("帧号: --")
            return
        self.push_fps_label.setText(
            f"推帧: {self._fps(self._push_win, now):.1f} fps")
        self.result_fps_label.setText(
            f"回结果: {self._fps(self._result_win, now):.1f} fps")
        self.latency_label.setText(
            "延迟: --" if self._last_latency_ms is None
            else f"延迟: {self._last_latency_ms:.0f} ms")
        self.frame_label.setText(
            "帧号: --" if self._push_idx < 0 else f"帧号: {self._push_idx}")

    def _refresh_controls(self) -> None:
        idle = not self._session_active and not self._creating
        self.start_btn.setEnabled(idle)
        self.stop_btn.setEnabled(self._session_active)
        self.src_combo.setEnabled(idle)
        self.src_stack.setEnabled(idle)
        self.stride_spin.setEnabled(idle)
        self.annotate_btn.setEnabled(self._session_active)
        # 快捷键仅会话期间启用, 避免吃输入框按键
        for sc in (self._sc_space, self._sc_p, self._sc_b):
            sc.setEnabled(self._session_active)
        # 删除类按钮仅标注态可用
        for b in (self.undo_point_btn, self.clear_box_btn, self.clear_group_btn):
            b.setEnabled(self._session_active and self._annotating)
        self.reset_btn.setEnabled(self._session_active)

    # ================= 退出 =================
    def shutdown(self) -> None:
        """页面清理: 停采集 -> 关 worker -> 归还会话。"""
        self._closing = True
        self._status_timer.stop()
        self._stop_session()

"""后台线程封装: 一次性 HTTP 调用(ApiCallWorker) + 视频 WS 会话驱动(VideoSessionWorker)。"""

from __future__ import annotations

import queue
from typing import Callable, Optional

from PySide6.QtCore import QThread, Signal

from GazeSystem_v1.api.client import Sam3Client


class ApiCallWorker(QThread):
    """一次性后台调用(HTTP): run 里执行 fn(*args, **kwargs)。"""

    succeeded = Signal(object)  # fn 返回值
    failed = Signal(str)        # str(异常)

    def __init__(self, fn: Callable, args: tuple = (),
                 kwargs: Optional[dict] = None, parent=None):
        super().__init__(parent)
        self._fn = fn
        self._args = args
        self._kwargs = kwargs or {}

    def run(self) -> None:
        try:
            self.succeeded.emit(self._fn(*self._args, **self._kwargs))
        except Exception as e:  # 后台调用一律回投错误, 不让线程死
            self.failed.emit(str(e))


# 持有运行中的 worker 引用, 防止被 GC(finished 后释放)
_active_workers: set = set()


def run_api(fn: Callable, *args, on_ok=None, on_err=None, parent=None,
            **kwargs) -> ApiCallWorker:
    """创建并启动一次性 API 调用 worker, 连接回调后返回该 worker。"""
    worker = ApiCallWorker(fn, args, kwargs, parent=parent)
    if on_ok is not None:
        worker.succeeded.connect(on_ok)
    if on_err is not None:
        worker.failed.connect(on_err)
    _active_workers.add(worker)
    worker.finished.connect(lambda: _active_workers.discard(worker))
    worker.start()
    return worker


# 队列条目类型
_KIND_PUSH = "push_frame"
_KIND_CMD = "cmd"
_KIND_SUBMIT = "submit"
_KIND_STOP = "stop"


def _is_fatal(exc: Exception) -> bool:
    """连接级致命错误(WS 连接断开等)判定; 服务端业务错误(ApiError)不算。"""
    mod = type(exc).__module__
    return mod.startswith("websockets") or isinstance(exc, (OSError, EOFError))


class VideoSessionWorker(QThread):
    """视频 WS 通道驱动器(一个视频会话一个实例)。

    UI 线程调下面的公共方法(内部入队串行执行); 事件经 signal 回投。
    """

    frame_result = Signal(dict)          # push_frame 的结果事件(含 mask_images: {gid: PIL.Image})
    command_result = Signal(str, dict)   # (action 名, 结果事件)
    propagate_event = Signal(dict)       # submit 事件流(propagate_start/prompts_applied/keyframe/propagate_done/cancelled)
    failed = Signal(str)
    closed = Signal()

    def __init__(self, client: Sam3Client, session_id: str, parent=None):
        super().__init__(parent)
        self._client = client
        self._session_id = session_id
        self._channel = None  # worker 线程持有; cancel() 可跨线程直接调
        self._queue: queue.Queue = queue.Queue()
        self._stopping = False

    # ---------------- 公共方法(UI 线程调用, 入队即返回) ----------------
    def push_frame(self, pil_image) -> None:
        """流式推帧: 队列里未执行的旧推帧直接丢弃, 只保留最新一帧(防积压)。"""
        with self._queue.mutex:
            q = self._queue.queue
            dropped = sum(1 for item in q if item[0] == _KIND_PUSH)
            if dropped:
                kept = [item for item in q if item[0] != _KIND_PUSH]
                q.clear()
                q.extend(kept)
            q.append((_KIND_PUSH, pil_image))
            self._queue.unfinished_tasks += 1 - dropped
            self._queue.not_empty.notify()

    def add_point(self, group_id, x, y, label, frame_idx) -> None:
        self._enqueue_cmd("add_point", group_id, x, y, label, frame_idx)

    def add_box(self, group_id, x1, y1, x2, y2, frame_idx) -> None:
        self._enqueue_cmd("add_box", group_id, x1, y1, x2, y2, frame_idx)

    def delete_point(self, group_id, frame_idx, point_index=-1) -> None:
        self._enqueue_cmd("delete_point", group_id, frame_idx, point_index)

    def clear_box(self, group_id, frame_idx) -> None:
        self._enqueue_cmd("clear_box", group_id, frame_idx)

    def clear_group(self, group_id) -> None:
        self._enqueue_cmd("clear_group", group_id)

    def load_prompt_file(self, file_data, merge_mode="append") -> None:
        self._enqueue_cmd("load_prompt_file", file_data, merge_mode)

    def get_frame(self, frame_idx, compute_if_missing=False) -> None:
        self._enqueue_cmd("get_frame", frame_idx, compute_if_missing)

    def reset(self) -> None:
        self._enqueue_cmd("reset")

    def submit(self, start_frame=None, end_frame=None, num_frames=None) -> None:
        self._queue.put((_KIND_SUBMIT, start_frame, end_frame, num_frames))

    def cancel(self) -> None:
        """请求取消传播: 直接调 channel.cancel()(不入队, client.py 保证线程安全)。"""
        ch = self._channel
        if ch is not None:
            try:
                ch.cancel()
            except Exception:
                pass  # 通道已坏时取消无意义, 错误会经 failed/事件流反映

    def shutdown(self) -> None:
        """置停止标志 + 入队哨兵; run 结束时 close channel 并发 closed。"""
        self._stopping = True
        self._queue.put((_KIND_STOP,))

    # ---------------- worker 线程 ----------------
    def _enqueue_cmd(self, action: str, *args, **kwargs) -> None:
        self._queue.put((_KIND_CMD, action, args, kwargs))

    def run(self) -> None:
        try:
            channel = self._client.open_video(self._session_id)
        except Exception as e:  # 连接失败: failed + closed 并结束
            self.failed.emit(str(e))
            self.closed.emit()
            return
        self._channel = channel
        try:
            while not self._stopping:
                try:
                    item = self._queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                kind = item[0]
                if kind == _KIND_STOP:
                    break
                try:
                    if kind == _KIND_PUSH:
                        self.frame_result.emit(channel.push_frame(item[1]))
                    elif kind == _KIND_CMD:
                        _, action, args, kwargs = item
                        result = getattr(channel, action)(*args, **kwargs)
                        self.command_result.emit(action, result)
                    elif kind == _KIND_SUBMIT:
                        _, start, end, num = item
                        # 迭代事件生成器逐事件回投; 生成器结束即传播结束
                        for ev in channel.submit(start_frame=start,
                                                 end_frame=end,
                                                 num_frames=num):
                            self.propagate_event.emit(ev)
                            if self._stopping:
                                break
                except Exception as e:  # 单条命令出错: 回投后继续循环; 连接级致命错误则退出
                    self.failed.emit(str(e))
                    if _is_fatal(e):
                        break
        finally:
            self._channel = None
            try:
                channel.close()
            except Exception:
                pass
            self.closed.emit()

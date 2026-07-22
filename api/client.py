"""
api 层 client 端: 同步 Python SDK(PySide6 前端只依赖这个模块)
- 图像: requests 走 HTTP, predict 返回 {group_id: PIL 掩码}
- 视频: websockets.sync 走 WS(阻塞式收发; 建议包一层 QThread, 事件用 signal 投回主线程)

消息配对: 服务端对"含掩码的结果"连发两条消息 —— JSON 摘要(has_masks=true)
+ 二进制掩码包; recv_event() 负责合并成一个事件 dict, 掩码在 "mask_images" 里
线程安全: websockets.sync 支持一个线程阻塞 recv 的同时另一个线程 send,
所以 UI 线程可以在 QThread 阻塞收事件时直接调 cancel()
"""

import json
from typing import Dict, Generator, Optional

import requests
from PIL import Image

from .protocol import (encode_image, png_bytes_to_image, unpack_mask_bundle)


class ApiError(RuntimeError):
    """服务端返回的业务错误(HTTP 400 / WS error 事件)"""


def _check(resp: requests.Response) -> Dict:
    if resp.status_code != 200:
        try:
            detail = resp.json().get("error", resp.text)
        except Exception:
            detail = resp.text
        raise ApiError(f"HTTP {resp.status_code}: {detail}")
    return resp.json()


class Sam3Client:
    def __init__(self, base_url: str = "http://127.0.0.1:8000"):
        self.base_url = base_url.rstrip("/")
        self.ws_url = base_url.replace("http", "ws", 1).rstrip("/")

    # ---------- 模型状态 ----------
    def model_status(self) -> Dict:
        """GET /model/status: 各模型启用状态 + GPU 显存(前端连接健康检查用)"""
        return _check(requests.get(f"{self.base_url}/model/status"))

    # ---------- 图像(HTTP) ----------
    def create_image_session(self, image: Image.Image) -> str:
        r = requests.post(f"{self.base_url}/image/sessions",
                          data=encode_image(image, "PNG"))
        return _check(r)["session_id"]

    def add_point(self, sid: str, group_id: int, x: float, y: float, label: int) -> Dict:
        return _check(requests.post(f"{self.base_url}/image/sessions/{sid}/points",
                                    json={"group_id": group_id, "x": x, "y": y,
                                          "label": label}))

    def add_box(self, sid: str, group_id: int,
                x1: float, y1: float, x2: float, y2: float) -> Dict:
        return _check(requests.post(f"{self.base_url}/image/sessions/{sid}/boxes",
                                    json={"group_id": group_id, "x1": x1, "y1": y1,
                                          "x2": x2, "y2": y2}))

    def delete_point(self, sid: str, group_id: int, point_index: int = -1) -> Dict:
        return _check(requests.delete(f"{self.base_url}/image/sessions/{sid}/points",
                                      json={"group_id": group_id,
                                            "point_index": point_index}))

    def clear_box(self, sid: str, group_id: int) -> Dict:
        return _check(requests.delete(f"{self.base_url}/image/sessions/{sid}/boxes",
                                      json={"group_id": group_id}))

    def clear_group(self, sid: str, group_id: int) -> Dict:
        return _check(requests.post(
            f"{self.base_url}/image/sessions/{sid}/groups/{group_id}/clear"))

    @staticmethod
    def _decode_mask_response(resp: requests.Response) -> Dict[int, Image.Image]:
        if resp.status_code != 200:
            _check(resp)
        _, _, masks = unpack_mask_bundle(resp.content)
        return {gid: png_bytes_to_image(png) for gid, png in masks.items()}

    def delete_group(self, sid: str, group_id: int) -> Dict[int, Image.Image]:
        r = requests.delete(f"{self.base_url}/image/sessions/{sid}/groups/{group_id}")
        return self._decode_mask_response(r)

    def predict_image(self, sid: str) -> Dict[int, Image.Image]:
        r = requests.post(f"{self.base_url}/image/sessions/{sid}/predict")
        return self._decode_mask_response(r)

    def close_image_session(self, sid: str) -> Dict:
        return _check(requests.delete(f"{self.base_url}/image/sessions/{sid}"))

    def load_image_prompt_file(self, sid: str, file_data: Dict,
                               merge_mode: str = "append") -> Dict:
        return _check(requests.post(
            f"{self.base_url}/image/sessions/{sid}/prompt_file",
            json={"file_data": file_data, "merge_mode": merge_mode}))

    # ---------- 视频会话创建(HTTP) ----------
    def create_stream_session(self, frame_stride: int = 1,
                              auto_predict: bool = True) -> str:
        r = requests.post(f"{self.base_url}/video/sessions",
                          json={"frame_stride": frame_stride,
                                "auto_predict": auto_predict})
        return _check(r)["session_id"]

    def create_offline_session(self, video_path: str, frame_stride: int = 1,
                               auto_predict: bool = False) -> Dict:
        """上传整个视频文件, 返回 {"session_id", "num_frames", ...}"""
        with open(video_path, "rb") as f:
            data = f.read()
        r = requests.post(f"{self.base_url}/video/sessions/offline",
                          params={"frame_stride": frame_stride,
                                  "auto_predict": auto_predict},
                          data=data)
        return _check(r)

    def close_video_session(self, sid: str) -> Dict:
        return _check(requests.delete(f"{self.base_url}/video/sessions/{sid}"))

    # ---------- 视频交互(WebSocket) ----------
    def open_video(self, sid: str) -> "VideoChannel":
        from websockets.sync.client import connect  # 延迟导入, 只用图像时不强依赖
        return VideoChannel(connect(f"{self.ws_url}/ws/video/{sid}"))


class VideoChannel:
    """一个视频会话的 WS 通道(阻塞式, 一个会话一个; 建议在 QThread 里驱动)"""

    def __init__(self, ws):
        self.ws = ws

    # ---- 收发 ----
    def _send_cmd(self, cmd: Dict) -> None:
        self.ws.send(json.dumps(cmd))

    def _recv_one(self, timeout: Optional[float] = None) -> Dict:
        msg = self.ws.recv(timeout=timeout)
        if isinstance(msg, bytes):
            frame_idx, progress, masks = unpack_mask_bundle(msg)
            return {"_is_bundle": True, "frame_idx": frame_idx, "progress": progress,
                    "mask_images": {g: png_bytes_to_image(b) for g, b in masks.items()}}
        return json.loads(msg)

    def recv_event(self, timeout: Optional[float] = None) -> Dict:
        """收一个完整事件(JSON 摘要; has_masks 时自动合并紧跟的掩码包)"""
        ev = self._recv_one(timeout)
        if ev.get("_is_bundle"):
            raise ApiError("收到未配对的掩码包(协议失步)")
        if ev.get("has_masks"):
            bundle = self._recv_one(timeout)
            if not bundle.get("_is_bundle"):
                raise ApiError("摘要后未收到掩码包(协议失步)")
            ev = {**ev, "frame_idx": bundle["frame_idx"],
                  "progress": bundle["progress"],
                  "mask_images": bundle["mask_images"]}
            ev.pop("has_masks", None)
        if ev.get("type") == "error":
            raise ApiError(ev.get("message", "unknown error"))
        return ev

    # ---- 推帧 ----
    def push_frame(self, frame: Image.Image, fmt: str = "JPEG") -> Dict:
        """流式推一帧(二进制上行), 返回该帧结果事件"""
        self.ws.send(encode_image(frame, fmt))
        return self.recv_event()

    # ---- 提示操作(每个命令对应一个结果事件) ----
    def add_point(self, group_id: int, x: float, y: float, label: int,
                  frame_idx: int) -> Dict:
        self._send_cmd({"action": "add_point", "group_id": group_id,
                        "x": x, "y": y, "label": label, "frame_idx": frame_idx})
        return self.recv_event()

    def add_box(self, group_id: int, x1: float, y1: float,
                x2: float, y2: float, frame_idx: int) -> Dict:
        self._send_cmd({"action": "add_box", "group_id": group_id,
                        "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                        "frame_idx": frame_idx})
        return self.recv_event()

    def delete_point(self, group_id: int, frame_idx: int,
                     point_index: int = -1) -> Dict:
        self._send_cmd({"action": "delete_point", "group_id": group_id,
                        "frame_idx": frame_idx, "point_index": point_index})
        return self.recv_event()

    def clear_box(self, group_id: int, frame_idx: int) -> Dict:
        self._send_cmd({"action": "clear_box", "group_id": group_id,
                        "frame_idx": frame_idx})
        return self.recv_event()

    def clear_group(self, group_id: int) -> Dict:
        self._send_cmd({"action": "clear_group", "group_id": group_id})
        return self.recv_event()

    def load_prompt_file(self, file_data: Dict, merge_mode: str = "append") -> Dict:
        """从 prompt 文件批量导入多帧提示(只记账, 调 submit 才传播)"""
        self._send_cmd({"action": "load_prompt_file", "file_data": file_data,
                        "merge_mode": merge_mode})
        return self.recv_event()

    def get_frame(self, frame_idx: int, compute_if_missing: bool = False) -> Dict:
        self._send_cmd({"action": "get_frame", "frame_idx": frame_idx,
                        "compute_if_missing": compute_if_missing})
        return self.recv_event()

    def reset(self) -> Dict:
        self._send_cmd({"action": "reset"})
        return self.recv_event()

    # ---- 传播控制 ----
    def submit(self, start_frame: Optional[int] = None,
               end_frame: Optional[int] = None,
               num_frames: Optional[int] = None) -> Generator[Dict, None, None]:
        """
        提交分割申请, 返回事件生成器(在 QThread 里迭代, 逐事件发 signal)
        终止事件: propagate_done / cancelled / error(error 会抛 ApiError)
        """
        cmd = {"action": "submit"}
        if start_frame is not None:
            cmd["start_frame"] = start_frame
        if end_frame is not None:
            cmd["end_frame"] = end_frame
        if num_frames is not None:
            cmd["num_frames"] = num_frames
        self._send_cmd(cmd)
        while True:
            ev = self.recv_event()
            yield ev
            if ev.get("type") in ("propagate_done", "cancelled"):
                return

    def cancel(self) -> None:
        """请求取消(可从 UI 线程调, 不等 ack; cancelled 事件会出现在事件流里)"""
        self._send_cmd({"action": "cancel"})

    def close(self) -> None:
        self.ws.close()

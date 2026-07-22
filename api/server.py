"""
api 层 server 端: FastAPI 应用
- 图像走 HTTP(请求/响应, predict 返回二进制掩码包)
- 视频会话创建走 HTTP; 交互(推帧/提示/传播)走 WebSocket
  文本消息 = JSON 命令与事件; 二进制消息 = 掩码包(下行) / 图像帧(上行)

职责只有三件: 解析请求 -> 调服务层 -> 按 protocol 编码返回; 不写任何业务判断
线程模型:
- 事件循环线程: 收发 WS 消息, 永不直接跑 GPU
- 工作线程(run_in_executor): 所有服务层调用; submit 生成器在其中迭代,
  事件经 call_soon_threadsafe 投入发送队列(Queue 非线程安全, 必须走这一步)
- cmd_lock 保证同一连接上的命令按到达顺序串行执行(cancel/submit 除外)
"""

import argparse
import asyncio
import json
import logging
import os
import tempfile
from typing import Dict, List, Optional

import uvicorn
from fastapi import Body, FastAPI, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, Response
from PIL import Image

from ..business.service_layer import SAM3ServiceLayer
from .protocol import decode_image, pack_mask_bundle

logger = logging.getLogger(__name__)


def decode_video_bytes(data: bytes) -> List[Image.Image]:
    """视频文件字节 -> 全部帧(decord 只认文件路径, 先落临时文件)"""
    import decord
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        f.write(data)
        path = f.name
    try:
        vr = decord.VideoReader(path)
        batch = vr.get_batch(list(range(len(vr)))).asnumpy()
        return [Image.fromarray(frame) for frame in batch]
    finally:
        os.unlink(path)


def create_app(service: SAM3ServiceLayer) -> FastAPI:
    app = FastAPI(title="SAM3 Segmentation Server")

    @app.exception_handler(ValueError)
    async def value_error_handler(request, exc):
        # 服务层的参数/状态错误统一翻译成 400, 客户端拿到干净的消息
        return JSONResponse(status_code=400,
                            content={"success": False, "error": str(exc)})

    # ---------- 通用 ----------
    @app.get("/model/status")
    def model_status():
        return service.get_model_status()

    # ---------- 图像(HTTP) ----------
    @app.post("/image/sessions")
    async def image_create(body: bytes = Body(...)):
        sid = service.create_image_session(decode_image(body))
        return {"session_id": sid}

    @app.post("/image/sessions/{sid}/points")
    def image_add_point(sid: str, payload: Dict = Body(...)):
        return service.add_point_to_group(sid, payload["group_id"],
                                          payload["x"], payload["y"], payload["label"])

    @app.post("/image/sessions/{sid}/boxes")
    def image_add_box(sid: str, payload: Dict = Body(...)):
        return service.add_box_to_group(sid, payload["group_id"], payload["x1"],
                                        payload["y1"], payload["x2"], payload["y2"])

    @app.delete("/image/sessions/{sid}/points")
    def image_delete_point(sid: str, payload: Dict = Body(...)):
        return service.delete_image_point(sid, payload["group_id"],
                                          payload.get("point_index", -1))

    @app.delete("/image/sessions/{sid}/boxes")
    def image_clear_box(sid: str, payload: Dict = Body(...)):
        return service.clear_image_box(sid, payload["group_id"])

    @app.post("/image/sessions/{sid}/groups/{gid}/clear")
    def image_clear_group(sid: str, gid: int):
        return service.clear_group(sid, gid)

    def _mask_response(result: Dict) -> Response:
        """含掩码的服务层结果 -> 二进制掩码包响应"""
        return Response(
            content=pack_mask_bundle(0, result.get("group_ids", []),
                                     result.get("masks_tensor")),
            media_type="application/octet-stream")

    @app.delete("/image/sessions/{sid}/groups/{gid}")
    def image_delete_group(sid: str, gid: int):
        return _mask_response(service.delete_group(sid, gid))

    @app.post("/image/sessions/{sid}/predict")
    def image_predict(sid: str):
        return _mask_response(service.predict_image(sid))

    @app.delete("/image/sessions/{sid}")
    def image_close(sid: str):
        return service.close_image_session(sid)

    # ---------- 图像 prompt 文件导入 ----------
    @app.post("/image/sessions/{sid}/prompt_file")
    def image_load_prompt_file(sid: str, payload: Dict = Body(...)):
        file_data = payload["file_data"]
        merge_mode = payload.get("merge_mode", "append")
        return service.load_image_prompt_file(sid, file_data, merge_mode)

    # ---------- 视频会话创建(HTTP) ----------
    @app.post("/video/sessions")
    def video_create_stream(payload: Dict = Body(...)):
        """流式会话: 不传帧, 之后经 WS 逐帧推入"""
        sid = service.create_video_session(
            video_frames=None,
            frame_stride=payload.get("frame_stride", 1),
            auto_predict=payload.get("auto_predict", True))
        return {"session_id": sid, "is_streaming": True}

    @app.post("/video/sessions/offline")
    async def video_create_offline(body: bytes = Body(...),
                                   frame_stride: int = Query(1),
                                   auto_predict: bool = Query(False)):
        """
        离线会话: 整个视频文件一次上传, 服务端解码全部帧

        TODO(v2): 全帧解码驻留内存有硬天花板 —— 实测 46305 帧 x 1080p
        ≈ 280GB RAM, 直接 OOM。长视频需改硬盘缓存/惰性解码(decord 按帧
        随机访问, 只缓存近期帧), 现阶段只对短视频(几千帧内)可用
        """
        frames = decode_video_bytes(body)
        sid = service.create_video_session(
            video_frames=frames, frame_stride=frame_stride,
            auto_predict=auto_predict)
        return {"session_id": sid, "is_streaming": False, "num_frames": len(frames)}

    @app.delete("/video/sessions/{sid}")
    def video_close(sid: str):
        return service.close_video_session(sid)

    # ---------- 视频交互(WebSocket) ----------
    @app.websocket("/ws/video/{session_id}")
    async def video_ws(ws: WebSocket, session_id: str):
        await ws.accept()
        try:
            service._get_video_session(session_id)
        except ValueError:
            await ws.close(code=4404)
            return
        await VideoWsHandler(ws, service, session_id).run()

    return app


class VideoWsHandler:
    """单个视频 WS 连接的生命周期"""

    def __init__(self, ws: WebSocket, service: SAM3ServiceLayer, session_id: str):
        self.ws = ws
        self.service = service
        self.sid = session_id
        self.loop = asyncio.get_running_loop()
        self.send_q: asyncio.Queue = asyncio.Queue()
        self.cmd_lock = asyncio.Lock()       # 普通命令串行化(保持到达顺序)
        self.submit_running = False          # 同一连接同时只允许一个传播

    # ---- 发送(任何线程都能调: 一律经 call_soon_threadsafe 入队) ----
    def _emit_json(self, payload: Dict) -> None:
        self.loop.call_soon_threadsafe(self.send_q.put_nowait, ("json", payload))

    def _emit_error(self, exc: Exception) -> None:
        self._emit_json({"type": "error", "message": str(exc)})

    def _emit_result(self, result: Dict) -> None:
        """服务层结果 -> JSON 摘要(去掉 masks 张量) + 二进制掩码包(有 masks 时)"""
        result = dict(result)
        masks = result.pop("masks", None)
        groups = result.get("groups", [])
        frame_idx = result.get("frame_idx", 0)
        progress = result.get("progress", 1.0)
        result["has_masks"] = masks is not None  # 客户端据此决定是否再收一个掩码包
        self._emit_json(result)
        if masks is not None:
            bundle = pack_mask_bundle(frame_idx, groups, masks, progress)
            self.loop.call_soon_threadsafe(self.send_q.put_nowait, ("bin", bundle))

    async def _sender(self) -> None:
        try:
            while True:
                kind, payload = await self.send_q.get()
                if kind == "json":
                    await self.ws.send_text(json.dumps(payload, ensure_ascii=False))
                else:
                    await self.ws.send_bytes(payload)
        except (WebSocketDisconnect, RuntimeError, asyncio.CancelledError):
            pass

    # ---- submit 工作线程: 迭代生成器, 逐事件转发 ----
    def _run_submit(self, cmd: Dict) -> None:
        kwargs = {k: cmd[k] for k in ("start_frame", "end_frame", "num_frames")
                  if k in cmd}
        try:
            for ev in self.service.submit_video_prompts(self.sid, **kwargs):
                if "masks" in ev:
                    self._emit_result(ev)          # keyframe: JSON 摘要 + 掩码包
                else:
                    self._emit_json(ev)            # start/applied/cancelled/done
        except Exception as e:
            self._emit_error(e)
        finally:
            self.loop.call_soon_threadsafe(self._on_submit_done)

    def _on_submit_done(self) -> None:
        self.submit_running = False

    # ---- 命令分派(在工作线程里执行, 避免阻塞事件循环) ----
    def _dispatch(self, cmd: Dict) -> Dict:
        svc = self.service
        sid = self.sid
        action = cmd["action"]
        if action == "add_point":
            return svc.add_video_point(sid, cmd["group_id"], cmd["x"], cmd["y"],
                                       cmd["label"], cmd["frame_idx"])
        if action == "add_box":
            return svc.add_video_box(sid, cmd["group_id"], cmd["x1"], cmd["y1"],
                                     cmd["x2"], cmd["y2"], cmd["frame_idx"])
        if action == "delete_point":
            return svc.delete_video_point(sid, cmd["group_id"], cmd["frame_idx"],
                                          cmd.get("point_index", -1))
        if action == "clear_box":
            return svc.clear_video_box(sid, cmd["group_id"], cmd["frame_idx"])
        if action == "clear_group":
            return svc.clear_video_group(sid, cmd["group_id"])
        if action == "load_prompt_file":
            return svc.load_video_prompt_file(sid, cmd["file_data"],
                                              cmd.get("merge_mode", "append"))
        if action == "get_frame":
            return svc.get_video_frame_result(sid, cmd["frame_idx"],
                                              cmd.get("compute_if_missing", False))
        if action == "reset":
            return svc.reset_video_tracking(sid)
        raise ValueError(f"未知 action: {action}")

    async def _handle_command(self, cmd: Dict) -> None:
        action = cmd.get("action")
        if action == "cancel":
            # 只置位线程安全事件, 无需进 executor, 也不占 cmd_lock
            self._emit_json(self.service.cancel_video_propagate(self.sid))
            return
        if action == "submit":
            if self.submit_running:
                raise ValueError("已有传播在进行, 先 cancel 或等它结束")
            self.submit_running = True
            self.loop.run_in_executor(None, self._run_submit, cmd)
            return
        # 普通命令: 持锁串行, 在工作线程里调服务层
        async with self.cmd_lock:
            result = await self.loop.run_in_executor(None, self._dispatch, cmd)
        if "masks" in result:
            self._emit_result(result)
        else:
            self._emit_json(result)

    async def run(self) -> None:
        sender = asyncio.create_task(self._sender())
        try:
            while True:
                msg = await self.ws.receive()
                if msg["type"] == "websocket.disconnect":
                    break
                if msg.get("bytes") is not None:
                    # 二进制上行 = 推帧(JPEG/PNG 图像字节)
                    try:
                        frame = decode_image(msg["bytes"])
                        async with self.cmd_lock:
                            result = await self.loop.run_in_executor(
                                None, self.service.push_video_frame, self.sid, frame)
                        self._emit_result(result)
                    except Exception as e:
                        self._emit_error(e)
                    continue
                try:
                    await self._handle_command(json.loads(msg["text"]))
                except Exception as e:
                    self._emit_error(e)
        finally:
            # 断连即取消在途传播, 不让工作线程白烧 GPU
            self.service.cancel_video_propagate(self.sid)
            sender.cancel()


def main():
    parser = argparse.ArgumentParser(description="SAM3 分割服务")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model-path", default="/root/workspace/modelRepo/SAM3")
    parser.add_argument("--device", default="cuda:1")
    args = parser.parse_args()

    service = SAM3ServiceLayer(model_path=args.model_path, device=args.device,
                               enable_tracker=True, enable_video=True)
    uvicorn.run(create_app(service), host=args.host, port=args.port)


if __name__ == "__main__":
    main()

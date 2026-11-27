"""
流式视频会话服务: 逐帧推入 / 计数器门控 / 有界合并丢弃 / flow 观测上报

流式场景: 帧随时间到达(顺序追加), 追踪节奏由"有界合并缓冲(见 api/server.py)
+ 计数器门控"控制, 稳定性来自有界性而非反馈回路:
- 门控: frames_since_track >= frame_stride 才追踪(计数器制, 取代 %stride 取模,
  无相位问题); 其余帧丢弃复用并记账(gate_dropped_count)
- 提示帧豁免: 提示帧永远无条件入库追踪(_ensure_prompt_frame_pushed 契约不变)
- 观测: 到达间隔/追踪耗时 EMA 随结果上报(flow), 供客户端自适应推帧速率(v2);
  实测合并缓冲把 business 到达率节流成处理率, 服务端侧无动态降载空间(死信号),
  动态调速属于客户端职责
将来扩展: 乱序重排缓冲(依赖底层 dict 帧仓的随机写能力)
"""

import time
from typing import Dict, Optional

from PIL import Image

from ..models import VideoSession
from .base import VideoSessionServiceBase

_EMA_ALPHA = 0.2   # EMA 平滑系数(观测上报用)


class StreamingVideoService(VideoSessionServiceBase):
    """流式视频业务逻辑"""

    # ---- 流式推帧 ----
    def push_video_frame(self, session_id: str, frame: Image.Image,
                         client_idx: Optional[int] = None) -> Dict:
        """
        流式收帧: 门控追踪(track)或丢弃复用(drop+reuse), 显式帧号契约
        client_idx=None 为旧协议客户端: 按到达顺序编号(兼容期, 见 unpack_push_frame)
        """
        session = self._get_video_session(session_id)
        if not session.is_streaming:
            raise ValueError("非流式会话, 离线视频请用 submit/get_video_frame_result")
        session.received_frame_count += 1
        session.last_frame = frame
        if client_idx is None:
            client_idx = session.received_frame_count - 1  # 旧协议: 顺序编号
        session.latest_frame_idx = client_idx

        # 补丁4b 首帧记录视频尺寸(PIL size=(w,h), original_size=(h,w)), 供提示提交使用
        if session.original_size is None:
            session.original_size = (frame.size[1], frame.size[0])
            session.video_session["video_height"] = frame.size[1]
            session.video_session["video_width"] = frame.size[0]

        now = time.monotonic()
        self._update_arrival_ema(session, now)

        # ---- 计数器门控: 追踪 或 丢弃复用 ----
        session.frames_since_track += 1
        if session.anchor_frame is None or session.frames_since_track >= session.frame_stride:
            session.frames_since_track = 0
            t0 = time.monotonic()
            res = self._push_streaming_frame(session, frame, client_idx) # 入库+追踪
            self._update_proc_ema(session, time.monotonic() - t0)
            return {"frame_idx": client_idx, "keyframe": True,
                    **self._flow_stats(session),
                    **self._visible_result(session, client_idx, res)}

        # 门控丢弃: 不入库, 复用最近前序结果(丢弃入账, 可观测)
        session.gate_dropped_count += 1
        prev = self._latest_cached_before(session, client_idx)
        res = session.keyframe_results[prev] if prev is not None else {"masks": None, "groups": []}
        return {"frame_idx": client_idx, "keyframe": False, "reused_from": prev,
                **self._flow_stats(session),
                **self._visible_result(session, client_idx, res)}

    # ---- flow 观测(EMA 上报, 供客户端自适应推帧速率; 服务端不做动态降载) ----
    def _update_arrival_ema(self, session: VideoSession, now: float) -> None:
        """到达间隔 EMA -> 输入速率(与客户端阻塞/非阻塞推帧无关, 只测量真实到达)"""
        if session.last_arrival_ts is not None:
            interval_ms = (now - session.last_arrival_ts) * 1000.0
            session.ema_arrival_ms = (interval_ms if session.ema_arrival_ms is None
                                      else _EMA_ALPHA * interval_ms
                                      + (1 - _EMA_ALPHA) * session.ema_arrival_ms)
        session.last_arrival_ts = now

    def _update_proc_ema(self, session: VideoSession, proc_s: float) -> None:
        """追踪耗时 EMA -> 处理速率(只在 track 帧更新, 丢弃帧不代表追踪成本)"""
        proc_ms = proc_s * 1000.0
        session.ema_proc_ms = (proc_ms if session.ema_proc_ms is None
                               else _EMA_ALPHA * proc_ms
                               + (1 - _EMA_ALPHA) * session.ema_proc_ms)

    def _flow_stats(self, session: VideoSession) -> Dict:
        """随每次结果事件上报的流控观测数据(UI 状态栏/客户端自适应推帧的数据源)"""
        return {"flow": {
            "ema_arrival_ms": session.ema_arrival_ms,
            "ema_proc_ms": session.ema_proc_ms,
            "stride": session.frame_stride,
            "gate_dropped": session.gate_dropped_count,
        }}

    def _push_streaming_frame(self, session: VideoSession, frame: Image.Image,
                              client_idx: int, track: bool = True) -> Dict:
        """把一帧推入底层 session(显式客户端帧号); track=True 时顺带完成该帧跟踪"""
        add_out = self.compute_engine.add_video_frames(session.video_session, [frame])
        sess_idx = add_out["frame_indices"][0] # sess_idx是底层SAM3推理会话里的帧号(连续, 与空洞无关)
        session.stream_c2s[client_idx] = sess_idx # 前端映射到后端
        session.stream_s2c[sess_idx] = client_idx # 后端映射到前端
        session.pushed_frame_count += 1
        if not track or not session.submitted_groups:
            # 0 物体防护: 底层 frame_idx 路径不查物体数, 空物体会 IndexError, 并且如果选择不做track, 直接跳过计算
            return self._cache_result(session, client_idx, None)
        out = self.compute_engine.predict_video_frame(session.video_session, sess_idx)
        return self._cache_result(session, client_idx, out["masks"])

"""
流式视频会话服务: 逐帧推入 / 网格关键帧 / 插帧复用

流式场景: 帧随时间到达(顺序追加), 网格帧推入即追踪, 非网格帧复用最近前序结果
将来扩展: 乱序重排缓冲(依赖底层 dict 帧仓的随机写能力)
"""

from typing import Dict

from PIL import Image

from ..models import VideoSession
from .base import VideoSessionServiceBase


class StreamingVideoService(VideoSessionServiceBase):
    """流式视频业务逻辑"""

    # ---- 流式推帧 ----
    def push_video_frame(self, session_id: str, frame: Image.Image) -> Dict:
        """流式收帧: 网格帧推入底层并track, 非关键帧复用最近结果(插帧)"""
        session = self._get_video_session(session_id)
        if not session.is_streaming:
            raise ValueError("非流式会话, 离线视频请用 submit/get_video_frame_result")
        session.received_frame_count += 1
        session.last_frame = frame

        # 补丁4b 首帧记录视频尺寸(PIL size=(w,h), original_size=(h,w)), 供提示提交使用
        if session.original_size is None:
            session.original_size = (frame.size[1], frame.size[0])
            session.video_session["video_height"] = frame.size[1]
            session.video_session["video_width"] = frame.size[0]


        client_idx = session.received_frame_count - 1

        # 为网格帧
        if session.anchor_frame is None or (client_idx - session.anchor_frame) % session.frame_stride == 0:
            res = self._push_streaming_frame(session, frame) # push了之后会有新的cache和keyframe_result

            # 网格帧推入即追踪，非网格提示帧由_ensure_prompt_frame_pushed 只推不算
            return {"frame_idx": client_idx, "keyframe": True,
                    **self._visible_result(session, client_idx, res)}

        prev = self._latest_cached_before(session, client_idx)
        res = session.keyframe_results[prev] if prev is not None else {"masks": None, "groups": []}
        return {"frame_idx": client_idx, "keyframe": False, "reused_from": prev,
                **self._visible_result(session, client_idx, res)}


    def _push_streaming_frame(self, session: VideoSession, frame: Image.Image,
                          track: bool = True) -> Dict:
        """把最近收到的一帧推入底层 session; track=True 时顺带完成该帧跟踪"""
        client_idx = session.received_frame_count - 1 # 帧号从0开始计数, received_frame_count是总的接受帧数, 是从1开始计数的
        add_out = self.compute_engine.add_video_frames(session.video_session, [frame])
        sess_idx = add_out["frame_indices"][0] # sess_idx是底层SAM3推理会话里的帧号
        session.stream_c2s[client_idx] = sess_idx # 前端映射到后端
        session.stream_s2c[sess_idx] = client_idx # 后端映射到前端
        session.pushed_frame_count += 1
        if not track or not session.submitted_groups:
            # 0 物体防护: 底层 frame_idx 路径不查物体数, 空物体会 IndexError, 并且如果选择不做track, 直接跳过计算
            return self._cache_result(session, client_idx, None)
        out = self.compute_engine.predict_video_frame(session.video_session, sess_idx)
        return self._cache_result(session, client_idx, out["masks"])

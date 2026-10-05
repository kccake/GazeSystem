"""会话管理器: 统一管理图像/视频会话生命周期(注册/查询/删除/过期清理)"""

import time
import uuid
import threading
import logging
from typing import Dict, Optional

from PIL import Image

from .models import ImageSession, VideoSession

logger = logging.getLogger(__name__)


class SessionManager:
    """
    统一管理所有会话生命周期
    把之前的session的生命周期管理直接提出来一个类
    """

    def __init__(self, max_age_seconds: float = 3600):
        self.max_age_seconds = max_age_seconds
        self.image_sessions: Dict[str, ImageSession] = {}
        self.video_sessions: Dict[str, VideoSession] = {}
        self._lock = threading.Lock()
        self._start_cleanup_timer()

    # 这个是后台用来定期清理线程的, 但是不知道怎么用, 以及是否鲁棒
    def _start_cleanup_timer(self):
        def cleanup_loop():
            while True:
                time.sleep(60)
                try:
                    self.cleanup_expired()
                except Exception as e:
                    logger.error(f"Cleanup error: {e}")

        thread = threading.Thread(target=cleanup_loop, daemon=True)
        thread.start()

    def register_image_session(self, image: Image.Image) -> str:
        session_id = str(uuid.uuid4())
        with self._lock:
            self.image_sessions[session_id] = ImageSession(
                session_id=session_id, image=image,
            )
        logger.info(f"注册图像会话: {session_id}")
        return session_id

    def get_image_session(self, session_id: str) -> Optional[ImageSession]:
        return self.image_sessions.get(session_id)

    def delete_image_session(self, session_id: str) -> bool:
        with self._lock: # 同一时刻，只有一个线程能拿到锁，进去执行；其他线程必须在外面排队等。
            if session_id in self.image_sessions:
                del self.image_sessions[session_id]
                return True
        return False

    def register_video_session(self, video_session: Dict, is_streaming: bool,
                               frame_stride: int = 1, auto_predict: bool = True) -> str:
        session_id = str(uuid.uuid4())
        with self._lock:
            self.video_sessions[session_id] = VideoSession(
                session_id=session_id, video_session=video_session,
                is_streaming=is_streaming,
                frame_stride=frame_stride, # 插帧步长, 1 表示逐帧计算掩码
                auto_predict=auto_predict, # 是否自动提交
                frame_count=video_session["num_frames"], # 或为冗余声明, 但是还是保留, 理论上应该从顶层拿到num_frames, 而不是从计算层
                original_size=((video_session["video_height"], video_session["video_width"])
                                if video_session["video_height"] is not None else None), # 作为冗余字段, 用起来方便
            )
        logger.info(f"注册视频会话: {session_id}, streaming={is_streaming}")
        return session_id

    def get_video_session(self, session_id: str) -> Optional[VideoSession]:
        return self.video_sessions.get(session_id)

    def delete_video_session(self, session_id: str) -> bool:
        with self._lock:
            session = self.video_sessions.get(session_id)
            if session is not None:
                store = session.video_session["session"].processed_frames
                if hasattr(store, "close"):
                    store.close()  # DiskFrameStore: 删除磁盘段文件, 当帧仓为disk时
                del self.video_sessions[session_id]
                return True
        return False

    def cleanup_expired(self):
        current_time = time.time()
        with self._lock:
            for sid in [s for s, v in self.image_sessions.items() if current_time - v.created_at > self.max_age_seconds]:
                del self.image_sessions[sid]
            for sid in [s for s, v in self.video_sessions.items() if current_time - v.created_at > self.max_age_seconds]:
                del self.video_sessions[sid]

    def get_stats(self) -> Dict:
        return {
            "image_sessions": len(self.image_sessions),
            "video_sessions": len(self.video_sessions),
        }

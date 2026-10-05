"""
业务服务门面: 组合图像/视频(离线/流式)子服务, 动态代理分发

对外接口与拆分前 SAM3ServiceLayer 完全一致(api/server.py 零改动):
- 门面自有: set_model / get_model_status / predict_text(占位)
- 其余方法经 _PROXY_MAP 代理到子服务实例

视频共享方法(add_video_point 等)挂 streaming 实例:
基类逻辑场景无关(内部分支依 session.is_streaming), 而 streaming 实例
持有 _push_streaming_frame(_ensure_prompt_frame_pushed 的流式路径需要)
"""

import threading
from typing import Dict

from ..compute import SAM3ComputeEngine
from .session_manager import SessionManager
from .image_service import ImageService
from .video.offline import OfflineVideoService
from .video.streaming import StreamingVideoService


class SAM3ServiceLayer:
    """SAM3 业务服务层(门面)"""

    def __init__(self, model_path: str = "/root/workspace/modelRepo/SAM3",
                 device: str = "cuda:1",
                 enable_tracker: bool = True,
                 enable_video: bool = False,
                 enable_text: bool = False,
                 compute_engine=None):
        # compute_engine 可注入(测试传 FakeEngine); 缺省时自建
        self.compute_engine = (compute_engine if compute_engine is not None
                               else SAM3ComputeEngine(
                                   model_path=model_path, device=device,
                                   enable_tracker=enable_tracker, enable_video=enable_video,
                                   enable_image=False)) # 这个是Engine在命名时的问题, 还没有改
        self.session_manager = SessionManager() # 创建一个Manager管理Session

        # 串行化所有GPU计算(单卡, 图像/视频共享), 这个后面会做优化的, 可以多GPU并行, 以及负载均衡的工作
        self._compute_lock = threading.Lock()

        # 子服务(共享 compute_engine / session_manager, 构造注入)
        self.image = ImageService(self.session_manager, self.compute_engine)
        self.offline_video = OfflineVideoService(self.session_manager, self.compute_engine)
        self.streaming_video = StreamingVideoService(self.session_manager, self.compute_engine)

        # 代理映射：方法名 -> (子服务属性名, 子服务方法名)
        self._PROXY_MAP = {
            # 图像分割
            "create_image_session": ("image", "create_image_session"),
            "close_image_session": ("image", "close_image_session"),
            "add_point_to_group": ("image", "add_point_to_group"),
            "add_box_to_group": ("image", "add_box_to_group"),
            "clear_group": ("image", "clear_group"),
            "delete_image_point": ("image", "delete_image_point"),
            "clear_image_box": ("image", "clear_image_box"),
            "delete_group": ("image", "delete_group"),
            "predict_image": ("image", "predict_image"),
            "predict_image_once": ("image", "predict_image_once"),
            "load_image_prompt_file": ("image", "load_image_prompt_file"),
            # 视频-离线专属
            "create_video_session": ("offline_video", "create_video_session"),
            "create_video_session_from_path": ("offline_video", "create_video_session_from_path"),
            "submit_video_prompts": ("offline_video", "submit_video_prompts"),
            "get_video_frame_result": ("offline_video", "get_video_frame_result"),
            "load_video_prompt_file": ("offline_video", "load_video_prompt_file"),
            # 视频-流式专属
            "push_video_frame": ("streaming_video", "push_video_frame"),
            # 视频-共享(base 逻辑, 挂 streaming 实例: 见模块 docstring)
            "_get_video_session": ("streaming_video", "_get_video_session"),
            "add_video_point": ("streaming_video", "add_video_point"),
            "add_video_box": ("streaming_video", "add_video_box"),
            "delete_video_point": ("streaming_video", "delete_video_point"),
            "clear_video_box": ("streaming_video", "clear_video_box"),
            "clear_video_group": ("streaming_video", "clear_video_group"),
            "cancel_video_propagate": ("streaming_video", "cancel_video_propagate"),
            "reset_video_tracking": ("streaming_video", "reset_video_tracking"),
            "close_video_session": ("streaming_video", "close_video_session"),
        }

    # ========== 门面自有方法 ==========
    def set_model(self, model_type: str, enabled: bool):
        # 这个enabled的命名并不好, 应该是emmm, 另一个名字, enabled应该是一种状态
        self.compute_engine.set_model(model_type, enabled)

    def get_model_status(self) -> Dict:
        return self.compute_engine.get_model_status()

    # TODO
    # 关于Text的还没有做
    def predict_text(self, image, text_prompt: str,
                     confidence_threshold: float = 0.5) -> Dict:
        """
        文本提示分割

        TODO:
        1. compute/text_engine.py 中 TextPromptEngine.predict() 需要完整实现
        2. 确认 Sam3Model / Sam3Processor 的 API 用法
        3. 统一返回格式为 torch.Tensor (num_objects, H, W)
        4. 添加输入校验
        5. 测试多物体文本分割
        """
        raise NotImplementedError(
            "文本提示分割 (predict_text) 尚未实现。\n"
            "需要完成的工作：\n"
            "1. compute/text_engine.py: TextPromptEngine.load() 确认模型加载\n"
            "2. compute/text_engine.py: TextPromptEngine.predict() 实现推理逻辑\n"
            "3. 确认返回格式: {'masks': torch.Tensor(N,H,W), 'scores': List[float], 'num_objects': int}\n"
            "4. 添加 _validate_text_prompt() 等输入校验\n"
            "5. 测试端到端流程"
        )

    # ========== 动态代理 ==========
    def __getattr__(self, name: str):
        """动态代理到子服务"""
        if name in self._PROXY_MAP:
            attr_name, method_name = self._PROXY_MAP[name]
            target = getattr(self, attr_name)
            return getattr(target, method_name)

        # 非代理方法，抛出 AttributeError
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

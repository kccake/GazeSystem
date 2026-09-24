"""
主计算引擎: 统一管理多个子引擎, 动态代理分发

SAM3ComputeEngine
├── 模型加载/卸载（通用）
├── 图像分割(ImageTrackerEngine)
│   └── predict()          ← 首次/增量推理统一入口
├── 视频分割(VideoTrackerEngine)
│   ├── init_session()     ← 初始化会话
│   ├── add_frame()    ← 流式只推帧(不追踪)
│   ├── add_prompt()       ← 交互式添加提示/文件添加提示
│   ├── predict_frame()    ← 单帧推理(流式/离线统一入口)
│   └── propagate()        ← 传播推理
└── 文本分割(TextPromptEngine)
    └── predict()
"""

import torch
from typing import Dict

from .image_engine import ImageTrackerEngine
from .video_engine import VideoTrackerEngine
from .text_engine import TextPromptEngine


class SAM3ComputeEngine:
    """统一管理多个子引擎"""

    def __init__(self,
                 model_path: str = "/root/workspace/modelRepo/SAM3",
                 device: str = "cuda:1",
                 enable_image: bool = False,
                 enable_tracker: bool = True,
                 enable_video: bool = False):
        """
        初始化 SAM3 计算引擎

        参数:
        - model_path: 模型路径
        - device: 计算设备，默认 "cuda:1"(优先 GPU)，可设为 "cpu"
        - enable_image: 是否启用文本分割模型(Sam3Model)
        - enable_tracker: 是否启用图像跟踪模型(Sam3TrackerModel)
        - enable_video: 是否启用视频跟踪模型(Sam3TrackerVideoModel)
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model_path = model_path

        # 子引擎（按需初始化并加载）
        self.image_tracker = None
        self.video_tracker = None
        self.text_prompt = None

        if enable_tracker:
            self.image_tracker = ImageTrackerEngine(self.device, model_path)
            self.image_tracker.load()
        if enable_video:
            self.video_tracker = VideoTrackerEngine(self.device, model_path)
            self.video_tracker.load()
        if enable_image:
            self.text_prompt = TextPromptEngine(self.device, model_path)
            self.text_prompt.load()

        # 代理映射：方法名 -> (引擎属性名, 引擎方法名)
        self._PROXY_MAP = {
            # 图像分割
            "predict_prompt": ("image_tracker", "predict"),
            # 视频分割
            "init_video_session": ("video_tracker", "init_session"),
            "add_video_frame": ("video_tracker", "add_frame"),
            "add_video_frames": ("video_tracker", "add_frames"),
            "add_video_prompt": ("video_tracker", "add_prompt"),
            "predict_video_frame": ("video_tracker", "predict_frame"),
            "propagate_video": ("video_tracker", "propagate"),
            "remove_video_object": ("video_tracker", "remove_object"),
            "remove_video_object_inputs": ("video_tracker", "remove_object_inputs"),
            "clear_video_objects": ("video_tracker", "clear_objects"),
            "close_video_session": ("video_tracker", "close_session"),
            # 文本分割
            "predict_text": ("text_prompt", "predict"),
        }

    def set_model(self, model_type: str, enabled: bool):
        """
        启用或禁用指定模型

        enabled=True: 初始化并加载到显存
        enabled=False: 从显存卸载并释放
        """
        import gc

        if model_type == "tracker":
            if enabled and self.image_tracker is None:
                self.image_tracker = ImageTrackerEngine(self.device, self.model_path)
                self.image_tracker.load()
            elif not enabled and self.image_tracker is not None:
                self.image_tracker.unload()
                del self.image_tracker
                self.image_tracker = None

        elif model_type == "video":
            if enabled and self.video_tracker is None:
                self.video_tracker = VideoTrackerEngine(self.device, self.model_path)
                self.video_tracker.load()
            elif not enabled and self.video_tracker is not None:
                self.video_tracker.unload()
                del self.video_tracker
                self.video_tracker = None

        elif model_type == "image":
            if enabled and self.text_prompt is None:
                self.text_prompt = TextPromptEngine(self.device, self.model_path)
                self.text_prompt.load()
            elif not enabled and self.text_prompt is not None:
                self.text_prompt.unload()
                del self.text_prompt
                self.text_prompt = None

        # 清理显存
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize(self.device)

    def get_model_status(self) -> Dict:
        """获取当前模型状态"""
        return {
            "image": {
                "enabled": self.text_prompt is not None,
            },
            "tracker": {
                "enabled": self.image_tracker is not None,
            },
            "video": {
                "enabled": self.video_tracker is not None,
            },
            "gpu_memory_gb": torch.cuda.memory_allocated() / 1024 ** 3 if torch.cuda.is_available() else 0
        }

    def __getattr__(self, name: str):
        """动态代理到子引擎"""
        if name in self._PROXY_MAP:
            attr_name, method_name = self._PROXY_MAP[name] # 这里的attr_name指的就是子Engine
            engine = getattr(self, attr_name)
            if engine is None:
                raise RuntimeError(f"{attr_name} 模型未启用")
            return getattr(engine, method_name) # 从子引擎去获取方法

        # 非代理方法，抛出 AttributeError
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

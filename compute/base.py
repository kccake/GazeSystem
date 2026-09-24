"""
引擎契约基类

自定义引擎(如魔改 SAM3)的约定:
- 自产引擎继承 BaseEngine, 获得生命周期契约(self.device/model_path/model/processor)
- 外部引擎/测试 mock 走鸭子类型, 不强制继承

各域引擎的原语清单(自定义引擎需与之一致才能挂进主引擎 _PROXY_MAP):
- 图像域(ImageTrackerEngine): predict()
- 视频域(VideoTrackerEngine): init_session / add_frame / add_frames /
  add_prompt / predict_frame / propagate / close_session /
  remove_object / remove_object_inputs / clear_objects
- 文本域(TextPromptEngine): predict()

注意: 与 SAM3 内部结构耦合的优化(如 _evict_old_output 滑窗逐出)
属于适配层实现细节, 留在各域引擎文件里, 不进本契约
"""

from abc import ABC, abstractmethod

import torch


class BaseEngine(ABC):
    """引擎生命周期契约: load/unload 对偶"""

    def __init__(self, device: torch.device, model_path: str):
        self.device = device
        self.model_path = model_path
        self.model = None
        self.processor = None

    @abstractmethod
    def load(self):
        """加载模型到设备"""

    @abstractmethod
    def unload(self):
        """卸载模型并释放资源"""
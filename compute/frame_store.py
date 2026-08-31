"""
磁盘帧仓: 把预处理后的视频帧(bf16 tensor)持久化到磁盘, 接口对齐 dict

背景: 离线长视频的全部预处理帧放内存会打爆 RAM(14315帧≈87GB),
且在页缓存紧张的宿主机上会触发直接回收 stall。本类把帧分段写成
safetensors 文件, 读帧时 mmap 零拷贝按需加载, RAM 占用只有一个小 buffer。
帧仓是在engine.init_session里随会话一起新创建的,目前帧仓无法复用,想要服用需要做会话持久化,
要做的有
1. 帧仓落盘(这个已经做了, 但是目的是为了缓解内存压力)
2. 会话状态落盘(包括提示、对象映射、追踪记忆、掩码等)
3. 身份映射：视频 → 帧仓目录（对视频路径/内容做 hash），打开时查"这个视频有没有现成的仓"

对接 Sam3TrackerVideoInferenceSession.processed_frames 的全部用法:
- add_new_frame: store[idx] = tensor   (bf16, CPU, 顺序追加)
- get_frame:     store[idx] -> tensor  (随后 .to(inference_device))
- num_frames:    len(store)

safetensors 文件不可追加, 所以分段存储: 内存 buffer 攒满 SEGMENT_FRAMES
(或 flush() 被调用)时序列化一段, 段索引留在内存。
"""
import os
import tempfile
import uuid

import torch
from safetensors.torch import save_file, safe_open


class DiskFrameStore:
    """
    dict-like 磁盘帧仓: {frame_idx: (C,H,W) bf16 tensor}, 仅顺序追加
    SEGMENT_FRAMES =  每段帧数; buffer 峰值 ≈ 128 * 6MB ≈ 780MB
    """

    SEGMENT_FRAMES = 128

    def __init__(self, store_dir = None):
        self._dir = store_dir or os.environ.get(
            "GAZE_FRAME_STORE_DIR") or tempfile.gettempdir()
        os.makedirs(self._dir, exist_ok=True)
        self._prefix = os.path.join(
            self._dir, f"frames_{uuid.uuid4().hex[:12]}") # 将 UUID 转为 32 位十六进制字符串, 并截取前 12 个字符
        self._len = 0
        self._shape = None # (C, H, W), 首帧写入时确定
        self._buffer = [] # 待落盘的帧
        self._segments = [] # [(path, frame_count)], 按序, 段文件路径, frame_count => 这个段装了多少帧
        self._handles = {} # path -> safe_open 句柄(复用 mmap), _handles 是"已打开段文件"的缓存，key 是段文件路径，value 是 safe_open 返回的句柄对象
        self._closed = False # 标记这个帧仓是否被关闭, 初始为False, 调用close()后变为True, 不可逆

    # ---- dict 协议 ----, 让本类"表现得像字典"的方法
    def __len__(self) -> int:
        return self._len

    def __setitem__(self, frame_idx: int, value: torch.Tensor):
        # 让自定义对象能够支持中括号赋值语法, obj[key] = value
        if self._closed:
            raise RuntimeError("DiskFrameStore 已关闭")
        if frame_idx != self._len:
            raise IndexError("磁盘帧仓只支持顺序追加")
        if value.dtype != torch.bfloat16:
            value = value.to(torch.bfloat16)
        value = value.detach().cpu().contiguous()
        if self._shape is None:
            self._shape = tuple(value.shape)
        elif tuple(value.shape) != self._shape:
            raise ValueError(f"帧尺寸不一致: 期望 {self._shape}, 实际 {tuple(value.shape)}")
        self._buffer.append(value)
        self._len += 1
        if len(self._buffer) >= self.SEGMENT_FRAMES:
            # 攒满 128 帧就强制落入硬盘，不等别人来调 flush()
            # 但是 engine 的 add_frames 每批结束会调 flush()
            # 每攒 32 帧就主动落盘，根本到不了 128, 这样做只是为了安全
            self.flush()

    def __getitem__(self, frame_idx: int) -> torch.Tensor:
        if not 0 <= frame_idx < self._len:
            raise IndexError(f"帧 {frame_idx} 不存在 (共 {self._len} 帧)")
        
        seg_start = self._len - len(self._buffer) # 先把已经放入硬盘的帧数计算出来, 这个start是倒序的(idx更大)
        if frame_idx >= seg_start:
            return self._buffer[frame_idx - seg_start]
        # 再按段索引定位文件
        base = 0
        # 用循环是担心由于不同engine的flush大小不同, 无法直接访问, 所以进行顺序访问
        # 但是使用的时候基本不影响效率问题, 但是是否需要优化, 后续需要考虑
        for path, count in self._segments:
            if frame_idx < base + count:
                handle = self._handles.get(path)
                if handle is None:
                    # safe_open 内部是 mmap, get_tensor 零拷贝
                    handle = safe_open(path, framework="pt")
                    self._handles[path] = handle
                return handle.get_tensor(f"{frame_idx - base:06d}") 
            base += count
        raise IndexError(f"帧 {frame_idx} 段索引异常")  # 理论不可达

    # ---- 生命周期 ----
    # 设计出一个用于将内存存储的chunk持久化到硬盘所需要的函数
    def flush(self):
        """把 buffer 序列化为一个新段; 每个 chunk 入库后由 engine 调用"""
        if self._closed or not self._buffer:
            # 如果帧仓已经关闭, 或者是_buff为空or不存在
            return
        path = f"{self._prefix}_seg{len(self._segments):05d}.safetensors"
        save_file({f"{i:06d}": t for i, t in enumerate(self._buffer)}, path) # {(0,tensor0),(1,tensor1), ...}
        self._segments.append((path, len(self._buffer))) # 记录到_segment中
        self._buffer.clear() # 将buffer清除

    def close(self, delete: bool = True):
        """关闭并(默认)删除所有段文件, 因为目前frame_store的目的是缓解内存压力, 还不是会话持久化"""
        if self._closed:
            return
        if not delete:
            self.flush() # 持久化路径: 先把 buffer 尾帧落盘
         
        self._closed = True
        self._handles.clear()
        self._buffer.clear()
        if delete:
            for path, _ in self._segments:
                if os.path.exists(path):
                    os.remove(path)
            self._segments.clear()

    @property # 像访问属性一样去访问方法
    def path(self) -> str:
        return self._prefix  # 段文件前缀(测试断言用)

    

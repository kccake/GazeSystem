# business/ 业务层

会话管理、场景逻辑、状态维护。不碰模型——所有计算经 `compute/` 引擎原语。

## 模块职责

| 文件 | 职责 |
|---|---|
| `models.py` | 数据模型：`PointGroup`(一组点+标签=一个物体)、`VideoSession`/`ImageSession`(会话状态，提示按帧组织)、`ImagePromptFile`/`VideoPromptFile`(提示文件 pydantic 格式) |
| `session_manager.py` | 会话生命周期：注册/查询/删除/过期清理(持锁)。删除时释放帧仓段文件 |
| `image_service.py` | 图像域业务：会话/组操作/纯点-纯框-混合分组推理/`image_embeddings` 增量 refine/提示文件导入 |
| `video/base.py` | `VideoSessionServiceBase`：离线/流式**共享**逻辑——辅助方法(flush_dirty/缓存五件套/提示记录)、提示操作(增删点框，含补丁3缓存失效)、生命周期(cancel/reset/close)、**双场景方法**(create_video_session 与 submit_video_prompts，内部按 is_streaming 分支) |
| `video/offline.py` | 离线场景：视频文件建会话(decord 按段解码)、提示文件批量导入、结果查询(非关键帧复用最近前序) |
| `video/streaming.py` | 流式场景：推帧(网格判断→track 或插帧复用)、`_push_streaming_frame` 覆写 base stub |
| `service_layer.py` | `SAM3ServiceLayer` 门面：组合三个子服务，`_PROXY_MAP` 动态代理(与计算层同构)。**对外接口稳定的唯一承诺点**——api/server.py 只面对它 |

## 关键机制

- **门面代理**：门面 `_PROXY_MAP` 映射全部对外方法，`__getattr__` 分发。加新方法 = 子服务实现 + 映射表加一行。门面自有：set_model / get_model_status / predict_text(占位)
- **依赖注入**：`compute_engine` / `session_manager` 由门面创建，构造传入各子服务；子服务之间不互相引用。`compute_engine` 可构造时注入(测试传 FakeEngine)
- **共享方法归属**：add_video_point 等挂 streaming 实例——基类逻辑场景无关(内部依 session.is_streaming 分支)，而 streaming 实例持有 `_push_streaming_frame`(`_ensure_prompt_frame_pushed` 的流式路径需要)
- **双场景方法在 base**：create_video_session(传 None=流式)/submit_video_prompts(帧范围按场景取)对两种会话都合法，归 base 避免实例错位

## 场景差异一览

| | 离线 | 流式 |
|---|---|---|
| 帧来源 | 全量可用(文件/内存) | 逐帧到达 |
| 帧仓 | disk(safetensors 分段) | ram(dict) |
| 入库 | add_video_frames 批量(32) | add_video_frames([单帧]) |
| 追踪时机 | submit 传播计划(网格∪提示帧) | 网格帧推入即 track |
| 提示帧约束 | 任意帧 | 仅最新帧(_check_stream_prompt_frame) |
| 非关键帧 | 复用最近前序关键帧(查询时) | 复用最近前序关键帧(推帧时返回) |

## 扩展落点

- **眼动仪回放**：gaze 文件 → 注视检测 → 生成 VideoPromptFile 同形态数据 → `load_video_prompt_file` 导入(offline.py) → 用户确认 → submit。计算层零改动
- **乱序重排缓冲**(流式将来)：会话级缓冲，排好序后仍经 add_video_frames 入库
- **背压/流控**(V3.3)：随插帧策略一起定契约

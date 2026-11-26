# GazeSystem 架构（V2 结构成熟期）

SAM3 交互式视频/图像分割系统。四层架构：UI → API → 业务 → 计算，依赖单向向下。

## 分层总览

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ UI 层 (PySide6, 笔记本运行)                                                    │
│                                                                              │
│   app.py ──► main_window.py ──► pages/                                       │
│                    │              ├── 图像分割                                │
│                    │              ├── 视频离线标注  ◄── 眼动仪回放(V3.1 规划)    │
│                    │              └── 视频实时流                              │
│                    │                                                         │
│                 context.py ──► workers.py (QThread 收 WS 事件)                │
│                    │                                                         │
│                 Sam3Client ──────────┐  唯一出口, HTTP + websockets.sync      │
└────────────────────────────────────┼────────────────────────────────────────┘
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ API 层 (FastAPI + websockets, 服务器)                                          │
│                                                                              │
│   server.py              client.py (UI 侧用)         protocol.py             │
│   ├─ HTTP 路由(图像/离线会话) └─ HTTP 上传视频          └─ encode/decode        │
│   └─ WS 路由(推帧/submit/控制)  WS submit 事件流       mask bundle 打包      │
│         │                                                                    │
│         ▼  门面调用面 ~25 个方法                                               │
└─────────────────────────────────────────────────────────────────────────────┘
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 业务层 business/ (场景与状态管理, 不碰模型)            详见 business/README.md  │
│                                                                              │
│  service_layer.py        SAM3ServiceLayer 门面                                │
│  ┌────────────────────────────────────────────────┐  set_model/get_status   │
│  │  _PROXY_MAP 动态代理 (与计算层同构模式)           │  predict_text(占位)     │
│  └────────────────────────────────────────────────┘                         │
│       │              │                  │                                    │
│       ▼              ▼                  ▼                                    │
│  image_service   video/offline.py   video/streaming.py    models.py          │
│  (ImageService)  (OfflineVideo-     (StreamingVideo-     会话数据模型/        │
│                   Service)          Service)              提示文件格式        │
│       │              │                  │                   │                │
│       │         文件建会话/           推帧/网格/           session_manager.py │
│       │         提示文件导入/          插帧复用/           SessionManager      │
│       │         结果查询(复用前序)     (乱序缓冲-V3.3)     (注册/过期清理)      │
│       │              │                  │                   │                │
│       └──────────────┴──────────────────┴───────────────────┘                │
│                        │                                                     │
│              video/base.py  VideoSessionServiceBase                          │
│              共享辅助(flush_dirty/缓存五件套/提示记录)                          │
│              提示操作(增删点框, 含补丁3缓存失效)                                │
│              生命周期(cancel/reset/close)                                     │
│              双场景方法(create_video_session / submit_video_prompts)          │
└─────────────────────────────────────────────────────────────────────────────┘
                                     ▼  引擎原语 (BaseEngine 契约, 鸭子类型扩展)
┌─────────────────────────────────────────────────────────────────────────────┐
│ 计算层 compute/ (纯模型计算, 无业务概念)                                        │
│                                                                              │
│  main_engine.py     SAM3ComputeEngine 主引擎                                  │
│  ┌──────────────────────────────────────────┐  get_model_status            │
│  │  _PROXY_MAP 动态代理                       │                              │
│  └──────────────────────────────────────────┘                              │
│       │              │                │                                      │
│       ▼              ▼                ▼                                      │
│  image_engine    video_engine     text_engine (占位)      validators.py      │
│  (Sam3Tracker    (Sam3TrackerVideo                           输入校验          │
│   Model)         Model) ──┬── init_session(frame_store 装配)                 │
│       │                   ├── add_frame(s) ──► 批末 synchronize(竞态修复)      │
│  predict()                ├── add_prompt / predict_frame                     │
│       │                   ├── propagate / close_session                        │
│       │                   └── _evict_old_output(滑窗逐出, 适配层专属)            │
│       │                          │                                           │
│       │                   frame_store.py  DiskFrameStore                     │
│       │                   dict 协议(len/getitem/setitem) 鸭子类型               │
│       │                   safetensors 分段落盘, UUID 前缀, GC 兜底清理          │
│       │                          ▲                                           │
│  base.py BaseEngine(load/unload 生命周期契约)                                  │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 数据流

### 离线传播（视频离线标注页）

```
视频文件 ──HTTP上传──► server ──from_path──► business: decord 逐 chunk(32帧)
   ──add_video_frames──► 帧仓(disk, safetensors 分段)
用户加点 ──► frame_prompts 记录(dirty) ──提交──► submit_video_prompts 生成器:
   传播计划 = 网格帧 ∪ 提示帧(lead 补算提示进入记忆)
   ──逐帧──► predict_frame ──► keyframe_results 缓存
   ──WS keyframe 事件──► UI 画 mask
规则: 非关键帧复用最近前序关键帧; 加提示→补丁3缓存失效→增量重算
```

### 流式实时（视频实时流页）

```
播放帧 ──WS推帧──► push_video_frame ──► 网格帧?
   是: add_frames([单帧]) + predict_frame(track) ──► 返回 mask
   否: 复用最近前序关键帧 mask(插帧)
提示: 只能加在最新帧(_check_stream_prompt_frame), 补推机制保证提示帧必算
```

## 关键设计决策

| 决策 | 内容 | 理由 |
|---|---|---|
| 双层门面 | compute/business 各一个 `_PROXY_MAP` 动态代理 | 调用面稳定: server/UI 不受内部演进影响; 代价是静态不可见(V4 候选: Protocol 类型化) |
| 引擎原语场景无关 | add_frames/predict_frame 不分在线离线 | 差异在调用模式而非计算模式; 场景逻辑全部上移到业务层 |
| 业务层按场景拆分 | offline/streaming 子类 + base 共享 | 眼动仪/乱序缓冲等场景扩展有明确落点 |
| 帧仓鸭子类型 | DiskFrameStore 实现 dict 协议偷梁换柱 | 存储策略(ram/disk)对 SAM3 底层完全无感 |
| 适配层/契约分离 | `_evict_old_output` 等 SAM3 专属优化不进 BaseEngine | 对魔改 SAM3 的人: 框架只承诺通用生命周期, 优化自由实现 |
| 竞态修复 | add_frames 批末 `synchronize(pixel_values.device)` | CUDA 异步 D2H 拷贝的设备级同步, 队列替代不了 |
| 会话生命周期对偶 | init_session ↔ close_session(删段文件) + GC 兜底 | 磁盘资源确定性释放为主, GC 安全网为辅 |

## 版本演进

- **V1 快速验证期**: 四层建立、引擎原语解耦(add_frame/predict_frame)、前端可用; 遗留补丁式代码与单文件巨石
- **V2 结构成熟期**(当前): ③计算层拆包、④业务层按场景拆包、②存储策略(帧仓/瘦身/泄漏修复)、竞态根治(含双卡设备参数)、测试体系+AST 逐方法审计
- **V3 产品能力期**(规划): 眼动仪回放模式 → 插帧/复用策略 → 实时流体验(含背压) → 多 GPU; 需求驱动, 非必要不重构

## 已知边界 (V3 待办)

- 推帧无线程序列化(server 每帧 `run_in_executor(None, ...)`)——UI 推帧快于处理时帧序可能乱, 待验证
- `cleanup_expired` 过期清理不释放帧仓段文件(磁盘泄漏第二路径)
- `_compute_lock` 预留未用(⑤ 多 GPU 前置)
- 传播吞吐 ~13 fps(76ms/帧, 模型前向占 99.9%); torch.compile 已探索(vision_encoder 仅 1.15×), 备用杠杆: 降输入分辨率

## 扩展落点

| 需求 | 落点 |
|---|---|
| 眼动仪回放(V3.1) | gaze 文件→注视检测→VideoPromptFile→`load_video_prompt_file`(offline.py)→确认→submit; 计算层零改动 |
| 乱序重排缓冲 | streaming.py 会话级缓冲, 排序后经 add_frames 入库 |
| 背压/流控 | 随插帧策略(V3.2/3.3)定契约 |
| 自定义引擎 | 继承 BaseEngine 实现视频域原语清单(见 compute/base.py), 挂进主引擎 _PROXY_MAP |

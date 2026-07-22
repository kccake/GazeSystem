# GazeSystem_v1 后端已知问题清单

> 前端开发过程中发现的后端（business / api 层）问题。
> **结论先行：经 PySide6 前端（`GazeSystem_v1/ui/`）正常使用，以下问题均不会触发**——前端已逐项规避。
> 它们只会在绕过前端（脚本直接调 API、手写 prompt 文件）或个别边缘竞态下出现。
> 记录日期：2026-07-19。

## 1. `load_video_prompt_file` 对纯框组必崩

- **位置**：`business/service_layer.py` `load_video_prompt_file`（约 :701-703）
- **现象**：导入含纯框组（`"points": null`）的视频 prompt 文件时，`len(None)` 抛 `TypeError`，经 WS 错误事件返回，报错文案不友好（不是预期的 `ValueError`）。
- **根因**：`group_data.get("points", [])` 在键存在但值为 `null` 时返回 `None` 而非默认值 `[]`。`labels` 同理。
- **触发条件**：发送未经处理的原始 JSON——包括把 `assets/video_prompt.json`（组 1 即纯框组）原样发给服务端。**自己按格式导出的文件反而导不回去。**
- **前端规避**：`ui/pages/video_offline_page.py` 导入前把 `points/labels: null` 规范化为 `[]`。
- **建议修法**：改为 `group_data.get("points") or []`（labels 同理），一行。

## 2. 非传播期 `cancel` 的 WS 消息串话

- **位置**：`api/server.py` `_handle_command` 的 cancel 分支（约 :253-255）
- **现象**：任何时刻收到 `cancel` 都会立即回一个 ack JSON（无 `type` 字段），且不参与"JSON 摘要 + 掩码包"配对协议。
- **触发条件与后果**：
  - 传播进行中：ack 混入 submit 事件流，无害（前端忽略无 type 事件）。
  - **非传播期**：ack 滞留在 socket 中，下一条命令的 `recv_event()` 会把它错收为自己的结果，此后全部消息配对错位，表现为掩码不刷新或 `ApiError("协议失步")`，需重开会话恢复。
  - 已知窄竞态：传播刚好出错结束的瞬间点了取消。
- **前端规避**：仅在传播进行中提供"取消"按钮；关闭会话时的 cancel 紧随其后断开连接，ack 无人读取，无害。
- **建议修法**：`submit_running == False` 时不回 ack（或直接忽略 cancel），一行。

## 3. 图像 `clear_group` 残留 `session.masks`

- **位置**：`business/service_layer.py` 图像组清理路径（约 :415-439）
- **现象**：清空组只清提示不清已算出的 mask。下次 predict 之前调 `delete_group`，返回值里可能带着已清空组的残留 mask。
- **前端规避**：`ui/pages/image_page.py` 清空组时本地同步摘除该组 mask；且对服务端返回的 masks 按"无本地提示的组不显示"过滤。
- **建议修法**：清组时同步清理 `session.masks`/`session.group_ids` 中对应行；不改也可接受（影响仅限直接调 API 的脚本）。

## 4. `get_frame(compute_if_missing=True)` 同步长计算阻塞 WS

- **位置**：`business/service_layer.py` `get_video_frame_result`
- **现象**：`compute_if_missing=True` 时在 WS 命令路径上同步跑完整段传播计算，长视频会长时间堵住该连接的所有其他命令。
- **前端规避**：前端所有 `get_frame` 调用恒用 `compute_if_missing=False`。
- **建议修法**：无需改代码，在 `api.md` 中加一句警告即可。

## 5. 流式会话加载 prompt 文件绕过最新帧检查

- **位置**：`business/service_layer.py` `load_video_prompt_file` 入口（对照 `_check_stream_prompt_frame`，约 :947-950）
- **现象**：流式会话中交互式加提示有"只能标最新收到的帧"校验，但 `load_prompt_file` 路径绕过了该校验；文件引用未推送过的帧时，flush 阶段 `_to_session_idx` 查映射可能抛 `KeyError`（不是友好的 `ValueError`）。
- **前端规避**：实时流页（`video_stream_page.py`）不提供 prompt 文件导入功能；离线会话 `is_streaming=False` 不受影响。
- **建议修法**：`load_video_prompt_file` 入口对 `session.is_streaming` 直接拒绝或复用 `_check_stream_prompt_frame`，一行。

---

## 附：前端对后端行为的其他依赖（后端改动时需同步检查前端）

- `ui/pages/video_stream_page.py` 把错误文案 **"没有已计算的关键帧"** 按字符串匹配视为良性落空（删光提示后 get_frame 的返回路径）。后端若改该文案，前端需同步。
- 图像页依赖 `predict_image` 在无任何提示时返回错误（"没有可用的提示"）这一行为，前端已在本地判空跳过该请求。
- `api.md` 中 prompt 文件格式说明已补（见 `service_layer.py:582-586` 附近 TODO 的 done 标记），后续改格式需同步 `assets/image_prompt.json`、`assets/video_prompt.json` 与 `ui/prompt_file.py` 的读写逻辑。

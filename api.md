# GazeSystem API 文档

> 后端服务基于 FastAPI 构建，提供 SAM3 模型的远程推理能力。
> 基础地址: `http://<host>:<port>` (默认端口 8000)

---

## 目录

1. [服务状态接口](#1-服务状态接口)
2. [模型管理 API](#2-模型管理-api)
3. [推理接口 - 点击分割](#3-推理接口---点击分割)
4. [推理接口 - 文本分割](#4-推理接口---文本分割)
5. [推理接口 - 框分割](#5-推理接口---框分割)
6. [文件上传接口（备用）](#6-文件上传接口备用)

---

## 1. 服务状态接口

### 1.1 根路径

- **URL**: `GET /`
- **说明**: 获取服务基本信息

**响应示例**:
```json
{
  "message": "SAM3 推理服务运行中",
  "gpu_available": true,
  "gpu_name": "NVIDIA GeForce RTX 4090",
  "model_loaded": true
}
```

### 1.2 健康检查

- **URL**: `GET /health`
- **说明**: 检查服务及各模型加载状态

**响应示例**:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "gpu_available": true,
  "gpu_name": "NVIDIA GeForce RTX 4090",
  "models": {
    "image_text": false,
    "tracker": true,
    "video": false
  }
}
```

---

## 2. 模型管理 API

### 2.1 获取模型状态

- **URL**: `GET /models/status`
- **说明**: 查看当前各模型的启用/加载状态及显存占用

**响应示例**:
```json
{
  "success": true,
  "image": {
    "enabled": false,
    "loaded": false
  },
  "tracker": {
    "enabled": true,
    "loaded": true
  },
  "video": {
    "enabled": false,
    "loaded": false
  },
  "gpu_memory_gb": 4.32
}
```

---

### 2.2 配置模型加载策略

- **URL**: `POST /models/configure`
- **说明**: 设置下次加载时要启用的模型（不立即加载/卸载）

**请求参数**:

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `enable_image` | bool | 否 | 是否启用 Image Text 模型 |
| `enable_tracker` | bool | 否 | 是否启用 Tracker 模型 |
| `enable_video` | bool | 否 | 是否启用 Video 模型 |

**请求示例**:
```json
{
  "enable_image": true,
  "enable_tracker": true,
  "enable_video": false
}
```

**响应示例**:
```json
{
  "success": true,
  "message": "配置已更新",
  "config": {
    "image": true,
    "tracker": true,
    "video": false
  }
}
```

---

### 2.3 动态加载单个模型

- **URL**: `POST /models/load`
- **说明**: 立即加载指定模型到显存

**请求参数**:

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `model_type` | string | 是 | 模型类型：`image` / `tracker` / `video` |

**请求示例**:
```json
{
  "model_type": "image"
}
```

**响应示例**:
```json
{
  "success": true,
  "message": "image 模型加载成功",
  "image": {
    "enabled": true,
    "loaded": true
  },
  "tracker": {
    "enabled": true,
    "loaded": true
  },
  "video": {
    "enabled": false,
    "loaded": false
  },
  "gpu_memory_gb": 8.15
}
```

---

### 2.4 动态卸载单个模型

- **URL**: `POST /models/unload`
- **说明**: 立即卸载指定模型释放显存

**请求参数**:

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `model_type` | string | 是 | 模型类型：`image` / `tracker` / `video` |

**请求示例**:
```json
{
  "model_type": "video"
}
```

**响应示例**:
```json
{
  "success": true,
  "message": "video 模型已卸载",
  "image": {
    "enabled": true,
    "loaded": true
  },
  "tracker": {
    "enabled": true,
    "loaded": true
  },
  "video": {
    "enabled": false,
    "loaded": false
  },
  "gpu_memory_gb": 4.32
}
```

---

### 2.5 重新加载所有模型

- **URL**: `POST /models/reload`
- **说明**: 先卸载所有模型，再按当前配置重新加载

**请求示例**: 无需 Body

**响应示例**:
```json
{
  "success": true,
  "message": "模型已重新加载",
  "image": {
    "enabled": true,
    "loaded": true
  },
  "tracker": {
    "enabled": true,
    "loaded": true
  },
  "video": {
    "enabled": false,
    "loaded": false
  },
  "gpu_memory_gb": 8.15
}
```

---

### 2.6 手动加载模型（兼容旧版）

- **URL**: `POST /load_models`
- **说明**: 手动触发模型加载，可指定模型路径

**请求参数**（Query）:

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `model_path` | string | 否 | 模型路径，默认 `/root/workspace/modelRepo/SAM3` |

**响应示例**:
```json
{
  "success": true,
  "message": "SAM3 模型加载成功"
}
```

---

## 3. 推理接口 - 点击分割

- **URL**: `POST /predict/click`
- **说明**: 通过点击坐标（正/负样本点）进行图像分割，使用 `Sam3TrackerModel`

**请求参数**:

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `image_base64` | string | 是 | Base64 编码的图片数据 |
| `click_points` | List[(x,y)] | 是 | 点击坐标数组，像素坐标，如 `[[500.0, 300.0]]` |
| `click_labels` | List[int] | 是 | 标签数组，`1`=正样本(前景)，`0`=负样本(背景)，如 `[1]` |
| `width` | int | 是 | 图片宽度 |
| `height` | int | 是 | 图片高度 |

**请求示例**:
```json
{
  "image_base64": "/9j/4AAQSkZJRgABAQEASABIAAD...",
  "click_points": [[500.0, 300.0]],
  "click_labels": [1],
  "width": 1000,
  "height": 652
}
```

**响应参数**:

| 字段 | 类型 | 说明 |
|------|------|------|
| `success` | bool | 是否成功 |
| `mask` | List[List[int]] | 二值掩码数据（0 或 255），二维数组 |
| `shape` | List[int] | 掩码尺寸，如 `[652, 1000]` |
| `message` | string | 状态信息 |
| `processing_time_ms` | float | 处理耗时（毫秒） |

**响应示例**:
```json
{
  "success": true,
  "mask": [
    [0, 0, 0, 255, 255, 0, ...],
    [0, 0, 255, 255, 255, 255, ...],
    ...
  ],
  "shape": [652, 1000],
  "message": "点击分割完成",
  "processing_time_ms": 245.67
}
```

---

## 4. 推理接口 - 文本分割

- **URL**: `POST /predict/text`
- **说明**: 通过文本描述进行图像分割，使用 `Sam3Model`

**请求参数**:

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `image_base64` | string | 是 | Base64 编码的图片数据 |
| `text_prompt` | string | 是 | 文本描述，如 `"person"`、`"shoe"`、`"car"` |
| `confidence_threshold` | float | 否 | 置信度阈值，默认 `0.5` |

**请求示例**:
```json
{
  "image_base64": "/9j/4AAQSkZJRgABAQEASABIAAD...",
  "text_prompt": "person",
  "confidence_threshold": 0.5
}
```

**响应参数**:

| 字段 | 类型 | 说明 |
|------|------|------|
| `success` | bool | 是否成功 |
| `masks` | List[Object] | 多个掩码对象列表，每个包含 `id`、`mask`、`shape` |
| `scores` | List[float] | 每个掩码的置信度分数 |
| `num_objects` | int | 检测到的对象数量 |
| `message` | string | 状态信息 |
| `processing_time_ms` | float | 处理耗时（毫秒） |

**响应示例**:
```json
{
  "success": true,
  "masks": [
    {
      "id": 0,
      "mask": [
        [0, 0, 255, 255, ...],
        [0, 255, 255, 255, ...],
        ...
      ],
      "shape": [652, 1000]
    },
    {
      "id": 1,
      "mask": [
        [0, 0, 0, 0, ...],
        [0, 0, 255, 255, ...],
        ...
      ],
      "shape": [652, 1000]
    }
  ],
  "scores": [0.92, 0.78],
  "num_objects": 2,
  "message": "文本分割完成，找到 2 个对象",
  "processing_time_ms": 567.34
}
```

---

## 5. 推理接口 - 框分割

- **URL**: `POST /predict/box`
- **说明**: 通过矩形框进行图像分割，内部转换为框中心点调用点击分割，使用 `Sam3TrackerModel`

**请求参数**:

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `image_base64` | string | 是 | Base64 编码的图片数据 |
| `box_xywh` | List[float] | 是 | 矩形框 `[x, y, width, height]`，像素坐标 |

**请求示例**:
```json
{
  "image_base64": "/9j/4AAQSkZJRgABAQEASABIAAD...",
  "box_xywh": [480.0, 290.0, 110.0, 360.0]
}
```

**响应参数**:

| 字段 | 类型 | 说明 |
|------|------|------|
| `success` | bool | 是否成功 |
| `mask` | List[List[int]] | 二值掩码数据（0 或 255），二维数组 |
| `shape` | List[int] | 掩码尺寸 |
| `message` | string | 状态信息 |
| `processing_time_ms` | float | 处理耗时（毫秒） |

**响应示例**:
```json
{
  "success": true,
  "mask": [
    [0, 0, 0, 255, 255, 0, ...],
    [0, 0, 255, 255, 255, 255, ...],
    ...
  ],
  "shape": [652, 1000],
  "message": "点击分割完成",
  "processing_time_ms": 198.45
}
```

---

## 6. 文件上传接口（备用）

### 6.1 点击分割 - 文件上传方式

- **URL**: `POST /predict/click_file`
- **说明**: 通过文件上传方式进行点击分割，适用于不方便使用 Base64 的场景
- **Content-Type**: `multipart/form-data`

**请求参数**:

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `file` | File | 是 | 图片文件（JPEG/PNG 等） |
| `click_points` | string | 是 | JSON 字符串，如 `"[[500.0, 300.0]]"` |
| `click_labels` | string | 是 | JSON 字符串，如 `"[1]"` |
| `width` | int | 否 | 图片宽度，默认自动读取 |
| `height` | int | 否 | 图片高度，默认自动读取 |

**请求示例** (curl):
```bash
curl -X POST "http://localhost:8000/predict/click_file" \
  -F "file=@image.jpg" \
  -F "click_points=[[500.0, 300.0]]" \
  -F "click_labels=[1]" \
  -F "width=1000" \
  -F "height=652"
```

**响应示例**:
```json
{
  "success": true,
  "mask": [
    [0, 0, 0, 255, 255, 0, ...],
    [0, 0, 255, 255, 255, 255, ...],
    ...
  ],
  "shape": [652, 1000],
  "message": "点击分割完成",
  "processing_time_ms": 245.67
}
```

---

## 附录：数据模型定义

### ClickPromptRequest
```python
class ClickPromptRequest(BaseModel):
    image_base64: str
    click_points: List[Tuple[float, float]]  # [(x, y), ...]
    click_labels: List[int]                   # [1, 0, ...]
    width: int
    height: int
```

### TextPromptRequest
```python
class TextPromptRequest(BaseModel):
    image_base64: str
    text_prompt: str
    confidence_threshold: Optional[float] = 0.5
```

### BoxPromptRequest
```python
class BoxPromptRequest(BaseModel):
    image_base64: str
    box_xywh: List[float]  # [x, y, width, height]
```

### InferenceResponse
```python
class InferenceResponse(BaseModel):
    success: bool
    mask: Optional[List] = None          # 点击/框分割返回
    masks: Optional[List] = None         # 文本分割返回（多对象）
    scores: Optional[List[float]] = None # 置信度分数
    shape: Optional[List[int]] = None   # 掩码尺寸
    num_objects: Optional[int] = None   # 对象数量
    message: str
    processing_time_ms: Optional[float] = None
```

### ModelConfigRequest
```python
class ModelConfigRequest(BaseModel):
    enable_image: Optional[bool] = None
    enable_tracker: Optional[bool] = None
    enable_video: Optional[bool] = None
```

### ModelLoadRequest / ModelUnloadRequest
```python
class ModelLoadRequest(BaseModel):
    model_type: str  # "image", "tracker", "video"

class ModelUnloadRequest(BaseModel):
    model_type: str  # "image", "tracker", "video"
```

---

## 启动参数

```bash
python model_server.py [--port PORT] [--host HOST] [--model-path MODEL_PATH]
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--port` | `8000` | 服务端口 |
| `--host` | `0.0.0.0` | 绑定地址 |
| `--model-path` | `/root/workspace/modelRepo/SAM3` | 模型路径 |

也可通过环境变量配置：`PORT`、`HOST`。

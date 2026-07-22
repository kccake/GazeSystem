from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Tuple
import uvicorn
import torch
import numpy as np
from PIL import Image
import io
import base64
import logging
import sys
import time
import os
import uuid


from transformers import (
    Sam3Model, Sam3Processor,
    Sam3VideoModel, Sam3VideoProcessor,
    Sam3TrackerModel, Sam3TrackerProcessor
)

# ============ 配置 ============
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(title="SAM3 推理服务", description="为本地桌面程序提供 SAM3 远程模型推理能力")

# 允许跨域（本地开发需要）
# 把所有权限开到最大, 让前端在与后端不同的端口的情况下也可以访问
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============ SAM3 模型管理器 ============
# 模型管理器存在过重嫌疑, 事事都要管, 承担了一部分与计算无关的逻辑
class SAM3ModelManager:
    """管理 SAM3 模型的按需加载和推理"""
    
    def __init__(self):
        self.device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        self.model_loaded = False
        self.model_path = "/root/workspace/modelRepo/SAM3"
        
        # 三个模型（按需加载）
        self.img_model = None      # Sam3Model - 文本提示分割
        self.img_processor = None  # Sam3Processor
        self.trk_model = None      # Sam3TrackerModel - 点击/框提示分割
        self.trk_processor = None  # Sam3TrackerProcessor
        self.vid_model = None      # Sam3VideoModel - 视频分割
        self.vid_processor = None  # Sam3VideoProcessor
        
        # 配置：按需加载模型（默认只加载tracker，可通过API动态控制）
        self.enable_image = False   # 文本分割
        self.enable_tracker = True  # 点击/框分割
        self.enable_video = False   # 视频分割

        # 交互式分割会话管理
        self.interactive_sessions = {}
        # 会话格式: {
        #   session_id: {
        #       "image": PIL.Image,
        #       "click_points": [(x, y), ...],
        #       "click_labels": [1, 0, ...],
        #       "last_mask": np.array or None,
        #       "created_at": timestamp,
        #       "active": bool
        #   }
        # }
        
        logger.info(f"🖥️ 使用设备: {self.device}")
        logger.info(f"📋 模型配置: image={self.enable_image}, tracker={self.enable_tracker}, video={self.enable_video}")
    
    def _load_single_model(self, model_type: str):
        """加载单个模型到显存"""
        try:
            if model_type == "image" and self.img_model is None:
                logger.info("   ... 加载 Image Text Model")
                # Sam3Model不好使, 这个但是用sam3的框架是好使的
                self.img_model = Sam3Model.from_pretrained(self.model_path).to(self.device)
                self.img_processor = Sam3Processor.from_pretrained(self.model_path)
                self.enable_image = True
                return True
                
            elif model_type == "tracker" and self.trk_model is None:
                # 这个好使
                logger.info("   ... 加载 Image Tracker Model")
                self.trk_model = Sam3TrackerModel.from_pretrained(self.model_path).to(self.device)
                self.trk_processor = Sam3TrackerProcessor.from_pretrained(self.model_path)
                self.enable_tracker = True
                return True
                
            elif model_type == "video" and self.vid_model is None:
                # 这个不清楚
                logger.info("   ... 加载 Video Model")
                self.vid_model = Sam3VideoModel.from_pretrained(self.model_path).to(self.device, dtype=torch.bfloat16)
                self.vid_processor = Sam3VideoProcessor.from_pretrained(self.model_path)
                self.enable_video = True
                return True
            
            else:
                logger.info(f"   ⏭️ {model_type} 模型已加载或类型未知")
                return True
                
        except Exception as e:
            logger.error(f"❌ 加载 {model_type} 模型失败: {e}")
            import traceback
            traceback.print_exc() # 把报错信息打印出来
            return False
    
    def _unload_single_model(self, model_type: str):
        """卸载单个模型释放显存"""
        import gc
        
        if model_type == "image" and self.img_model is not None:
            logger.info("   ... 卸载 Image Text Model")
            del self.img_model
            del self.img_processor
            self.img_model = None
            self.img_processor = None
            self.enable_image = False
            
        elif model_type == "tracker" and self.trk_model is not None:
            logger.info("   ... 卸载 Image Tracker Model")
            del self.trk_model
            del self.trk_processor
            self.trk_model = None
            self.trk_processor = None
            self.enable_tracker = False
            
        elif model_type == "video" and self.vid_model is not None:
            logger.info("   ... 卸载 Video Model")
            del self.vid_model
            del self.vid_processor
            self.vid_model = None
            self.vid_processor = None
            self.enable_video = False
        
        # 强制垃圾回收和显存清理, 这样处理之后会把模型unload, 但不会删除掉Pytorch Cuda Context的固定显存开销
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        logger.info(f"📊 当前显存占用: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        return True
    
    def load_models(self, model_path: str = None):
        """按需加载 SAM3 模型到显存"""
        if model_path:
            self.model_path = model_path
            
        try:
            logger.info("⏳ 正在按需加载 SAM3 模型...")
            
            if self.enable_image:
                self._load_single_model("image")
            else:
                logger.info("   ⏭️ 跳过 Image Text Model (未启用)")
            
            if self.enable_tracker:
                self._load_single_model("tracker")
            else:
                logger.info("   ⏭️ 跳过 Image Tracker Model (未启用)")
            
            if self.enable_video:
                self._load_single_model("video")
            else:
                logger.info("   ⏭️ 跳过 Video Model (未启用)")
            
            self.model_loaded = True
            logger.info("✅ 模型加载完成！")
            return True
            
        except Exception as e:
            logger.error(f"❌ 模型加载失败: {e}")
            import traceback
            traceback.print_exc() # 把保存打印出来
            return False
    
    def get_model_status(self):
        """
        获取当前模型加载状态
        如果enabled == True并执行_load_single_model()成功,则loaded状态为True
        """
        return {
            "image": {
                "enabled": self.enable_image,
                "loaded": self.img_model is not None
            },
            "tracker": {
                "enabled": self.enable_tracker,
                "loaded": self.trk_model is not None
            },
            "video": {
                "enabled": self.enable_video,
                "loaded": self.vid_model is not None
            },
            "gpu_memory_gb": torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
        }
    
    def predict_click(self, image: Image.Image, click_points: List[Tuple[float, float]], 
                      click_labels: List[int], width: int, height: int) -> dict:
        """
        点击提示分割 - 使用 Sam3TrackerModel
        
        Args:
            image: PIL Image
            click_points: [(x, y), ...] 像素坐标
            click_labels: [1, 0, ...] 1=正样本(前景), -1=负样本(背景)
            width, height: 图片尺寸
        
        Returns:
            mask: 二值掩码 numpy array
            scores: 置信度分数
        """
        if not self.model_loaded or self.trk_model is None:
            raise HTTPException(status_code=503, detail="Tracker 模型尚未加载")
        
        # 构建输入格式: [[[x, y]]] -> [Batch, Point_Group, Point, Coord], 现在是没有办法做交互的
        # 并且点的逻辑是[image_dim, object_dim, point_per_object, coord_dim]
        points_formatted = [[[
            [float(x), float(y)] for x, y in click_points
        ]]]
        labels_formatted = [[[
            int(label) for label in click_labels
        ]]]
        
        # 预处理
        inputs = self.trk_processor(
            images=image,
            input_points=points_formatted,
            input_labels=labels_formatted,
            return_tensors='pt'
        ).to(self.device)
        
        # 推理
        with torch.no_grad():
            # SAM3的mask decoder默认输出3个mask + 3个对应的IoU分数, 当multimask_output=True时, 会把这三个mask都给你
            # 其中这三个mask1,2,3, 分别为最细粒度(包含更多小细节), 中等粒度, 最粗粒度(可能包含更多背景)
            outputs = self.trk_model(**inputs, multimask_output=False)
            
        # 后处理 - 获取掩码
        masks = self.trk_processor.post_process_masks(
            outputs.pred_masks.cpu(),
            inputs['original_sizes'],
            binarize=True
        )[0]
        
        # 提取第一个 mask (形状可能是 [1, H, W] 或 [H, W])
        mask_np = masks[0][0].numpy() if masks.ndim == 4 else masks[0].numpy()
        
        # 转换为 uint8 格式
        mask_uint8 = (mask_np * 255).astype(np.uint8)
        
        return {
            "mask": mask_uint8.tolist(), # 转化为列表, 可以塞到JSON里
            "shape": mask_uint8.shape,
            "num_points": len(click_points),
            "message": "点击分割完成"
        }
    
    # 这个暂时比较远, 先不去考虑
    def predict_text(self, image: Image.Image, text_prompt: str, 
                     confidence_threshold: float = 0.5) -> dict:
        """
        文本提示分割 - 使用 Sam3Model
        
        Args:
            image: PIL Image
            text_prompt: 文本描述，如 "person", "shoe"
            confidence_threshold: 置信度阈值
        
        Returns:
            masks: 多个掩码列表
            scores: 置信度分数列表
            bboxes: 边界框列表
        """
        if not self.model_loaded or self.img_model is None:
            raise HTTPException(status_code=503, detail="Image 模型尚未加载")
        
        # 创建 processor (每次新建以支持不同 threshold)
        processor = Sam3Processor(self.img_model, confidence_threshold=confidence_threshold)
        
        # 设置图片
        inference_state = processor.set_image(image)
        
        # 重置提示并设置文本提示
        processor.reset_all_prompts(inference_state)
        inference_state = processor.set_text_prompt(prompt=text_prompt, state=inference_state)
        
        # 提取结果
        masks = inference_state.get('masks', [])
        scores = inference_state.get('scores', [])
        
        # 转换 masks 为可序列化格式
        mask_list = []
        for i, mask in enumerate(masks):
            if hasattr(mask, 'cpu'):
                mask_np = mask.squeeze(0).cpu().numpy()
            else:
                mask_np = np.array(mask)
            mask_list.append({
                "id": i,
                "mask": (mask_np * 255).astype(np.uint8).tolist(),
                "shape": mask_np.shape
            })
        
        return {
            "masks": mask_list,
            "scores": [float(s) for s in scores],
            "num_objects": len(mask_list),
            "message": f"文本分割完成，找到 {len(mask_list)} 个对象"
        }
    
    # 这里的实现不是真正的用box的函数, 而是取中心点, 然后用点提示做的
    def predict_box(self, image: Image.Image, box_xywh: List[float]) -> dict:
        """
        框提示分割 - 使用 Sam3TrackerModel
        
        Args:
            image: PIL Image
            box_xywh: [x, y, width, height] 像素坐标
        
        Returns:
            mask: 分割掩码
        """
        if not self.model_loaded or self.trk_model is None:
            raise HTTPException(status_code=503, detail="Tracker 模型尚未加载")
        
        width, height = image.size
        
        # 转换为模型需要的格式 (cx, cy, w, h) 并归一化
        from sam3.model.box_ops import box_xywh_to_cxcywh
        from sam3.visualization_utils import normalize_bbox
        
        box_cxcywh = box_xywh_to_cxcywh(torch.tensor(box_xywh).view(-1, 4))
        norm_box = normalize_bbox(box_cxcywh, width, height).flatten().tolist()
        
        # 使用 processor 处理
        # 这里简化处理，实际可以使用 add_geometric_prompt
        # 为了简化，我们转换为点击点（框中心）来推理
        center_x = box_xywh[0] + box_xywh[2] / 2
        center_y = box_xywh[1] + box_xywh[3] / 2
        
        return self.predict_click(
            image, 
            [(center_x, center_y)], 
            [1], 
            width, 
            height
        )

    def start_interactive_session(self, image: Image.Image) -> str:
        """
        开始一个新的交互式分割会话
        
        Args:
            image: PIL Image
        
        Returns:
            session_id: 会话唯一标识
        """
        session_id = str(uuid.uuid4())
        self.interactive_sessions[session_id] = {
            "image": image,
            "click_points": [],
            "click_labels": [],
            "last_mask": None,
            "created_at": time.time(),
            "active": True
        }
        logger.info(f"交互式会话已创建: {session_id}")

        return session_id

    def add_point_and_predict(self, session_id: str, point: Tuple[float, float], 
                              label: int) -> dict:
        """
        向会话中添加一个点并执行分割
        
        Args:
            session_id: 会话ID
            point: (x, y) 像素坐标
            label: 1=正样本, 0=负样本
        
        Returns:
            分割结果字典
        """

        if session_id not in self.interactive_sessions:
            raise HTTPException(status_code=404, detail=f"会话 {session_id} 不存在")
        
        session = self.interactive_sessions[session_id]

        if not session['active']:
            raise HTTPException(status_code=400, detail=f"会话 {session_id} 已暂停")
        
        # 添加新点
        session['click_points'].append(point)
        session['click_labels'].append(label)

        # 使用所有点进行分割
        image = session['image']
        width, height = image.size

        result = self.predict_click(
            image=image,
            click_points=session['click_points'],
            click_labels=session['click_labels'],
            width=width,
            height=height
        )

        # 保存最新mask到会话
        mask_np = np.array(result["mask"])
        session['last_mask'] = mask_np

        result['num_points'] = len(session['click_points'])
        result['session_id'] = session_id
        
        logger.info(f"会话 {session_id}: 已添加点 {point} (label={label}), 总点数: {len(session['click_points'])}")

        return result

    def pause_interactive_session(self, session_id: str) -> dict:
        """暂停交互式会话"""
        # 暂停会话采用了后端session['active'] = False的逻辑, 这时add_point_and_predict就会无法使用, 没有在前端禁止接收数据

        if session_id not in self.interactive_sessions:
            raise HTTPException(status_code=404, detail=f"会话 {session_id} 不存在")
        
        self.interactive_sessions[session_id]["active"] = False
        logger.info(f"⏸️ 会话 {session_id} 已暂停")
        
        return {
            "success": True,
            "message": f"会话 {session_id} 已暂停",
            "session_id": session_id
        }
    
    def resume_interactive_session(self, session_id: str) -> dict:
        """恢复已暂停的交互式会话"""
        if session_id not in self.interactive_sessions:
            raise HTTPException(status_code=404, detail=f"会话 {session_id} 不存在")
        
        session = self.interactive_sessions[session_id]
        
        if session["active"]:
            return {
                "success": True,
                "message": f"会话 {session_id} 已经是活跃状态",
                "session_id": session_id
            }
        
        session["active"] = True
        logger.info(f"▶️ 会话 {session_id} 已恢复")
        
        return {
            "success": True,
            "message": f"会话 {session_id} 已恢复",
            "session_id": session_id
        }
    
    def clear_session_points(self, session_id: str) ->dict:
        """清除会话中的所有点，但保留会话和图片"""
        if session_id not in self.interactive_sessions:
            raise HTTPException(status_code=404, detail=f"会话 {session_id} 不存在")
        
        session = self.interactive_sessions[session_id]
        session["click_points"] = []
        session["click_labels"] = []
        session["last_mask"] = None
        session["active"] = True  # 清除点后自动恢复活跃

        logger.info(f"会话 {session_id} 的点已清除")

        return {
            "success": True,
            "message": f"会话 {session_id} 的点已清除",
            "session_id": session_id
        }

    def save_session_result(self, session_id: str, return_format: str = "mask") -> dict:
        """
        保存会话的最终分割结果
        
        Args:
            session_id: 会话ID
            return_format: "mask" 返回掩码数组, "overlay" 返回叠加图base64
        
        Returns:
            包含分割结果的字典
        """

        if session_id not in self.interactive_sessions:
            raise HTTPException(status_code=404, detail=f"会话 {session_id} 不存在")
        
        session = self.interactive_sessions[session_id]

        if session["last_mask"] is None:
            raise HTTPException(status_code=400, detail="会话尚未产生分割结果，请先添加点")
        
        mask = session["last_mask"]
        image = session["image"]

        result = {
            "success": True,
            "mask": mask.tolist(),
            "shape": list(mask.shape),
            "message": "分割结果已保存",
            "session_id": session_id
        }

        # 可选：生成叠加图
        if return_format == "overlay":
            overlay = self._create_overlay(image, mask)
            result["mask_base64"] = overlay
        
        logger.info(f"💾 会话 {session_id} 的结果已保存")

        return result
    
    def _create_overlay(self, image: Image.Image, mask: np.ndarray) -> str:
        """创建分割叠加图并返回 base64"""
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        ax.imshow(image)
        ax.imshow(mask, alpha=0.5, cmap='jet')
        ax.axis('off')
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        buf.seek(0)
        overlay_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)
        
        return overlay_base64
    
    def get_session_info(self, session_id: str) -> dict:
        """获取会话信息"""
        if session_id not in self.interactive_sessions:
            raise HTTPException(status_code=404, detail=f"会话 {session_id} 不存在")
        
        session = self.interactive_sessions[session_id]
        return {
            "session_id": session_id,
            "active": session["active"],
            "num_points": len(session["click_points"]),
            "has_result": session["last_mask"] is not None,
            "created_at": session["created_at"]
        }
    
    def cleanup_old_sessions(self, max_age_seconds: float = 3600):
        """清理过期的会话"""
        current_time = time.time()
        expired_sessions = []
        
        for session_id, session in self.interactive_sessions.items():
            if current_time - session["created_at"] > max_age_seconds:
                expired_sessions.append(session_id)
        
        for session_id in expired_sessions:
            del self.interactive_sessions[session_id]
            logger.info(f"🗑️ 过期会话已清理: {session_id}")
        
        return len(expired_sessions)

# 全局模型管理器, 用模型管理器来完成模型的加载和卸载, 并完成业务的逻辑
model_manager = SAM3ModelManager()

# ============ 数据模型 ============
# baseModel用于数据校验, 这里的图片写成了base64的前端方式, 没有通过后端来完成, 这里可以提升, 用后端来完成去提高兼容性
class ClickPromptRequest(BaseModel):
    image_base64: str
    click_points: List[Tuple[float, float]]  # [(x, y), ...]
    click_labels: List[int]                   # [1, 0, ...]
    width: int
    height: int

class TextPromptRequest(BaseModel):
    image_base64: str
    text_prompt: str
    confidence_threshold: Optional[float] = 0.5

class BoxPromptRequest(BaseModel):
    image_base64: str
    box_xywh: List[float]  # [x, y, width, height]

# ============ 交互式分割数据模型 ============
class InteractiveStartRequest(BaseModel):
    image_base64: str

class InteractiveStartResponse(BaseModel):
    success: bool
    session_id: Optional[str] = None
    message: str

class InteractiveAddPointRequest(BaseModel):
    session_id: str
    point: Tuple[float, float]  # (x, y)
    label: int = 1              # 1=正样本(前景), 0=负样本(背景)

class InteractiveAddPointResponse(BaseModel):
    success: bool
    mask: Optional[List] = None
    shape: Optional[List[int]] = None
    num_points: Optional[int] = None
    message: str

class InteractivePauseRequest(BaseModel):
    session_id: str

class InteractiveResumeRequest(BaseModel):
    session_id: str

class InteractiveClearRequest(BaseModel):
    session_id: str

class InteractiveSaveRequest(BaseModel):
    session_id: str
    return_format: Optional[str] = "mask"  # "mask" | "overlay"

class InteractiveSaveResponse(BaseModel):
    success: bool
    mask: Optional[List] = None
    mask_base64: Optional[str] = None
    shape: Optional[List[int]] = None
    message: str
################################################

class InferenceResponse(BaseModel):
    success: bool
    mask: Optional[List] = None
    masks: Optional[List] = None
    scores: Optional[List[float]] = None
    shape: Optional[List[int]] = None
    num_objects: Optional[int] = None
    message: str
    processing_time_ms: Optional[float] = None


# ============ 辅助函数 ============
def decode_image(image_base64: str) -> Image.Image:
    """解码 base64 图片"""
    image_data = base64.b64decode(image_base64)
    return Image.open(io.BytesIO(image_data)).convert("RGB")


# ============ API 路由 ============

@app.get("/")
async def root():
    return {
        "message": "SAM3 推理服务运行中",
        "gpu_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "model_loaded": model_manager.model_loaded
    }

@app.get("/health")
async def health_check():
    """健康检查接口"""
    return {
        "status": "healthy",
        "model_loaded": model_manager.model_loaded,
        "gpu_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "models": {
            "image_text": model_manager.img_model is not None,
            "tracker": model_manager.trk_model is not None,
            "video": model_manager.vid_model is not None
        }
    }

# ============ 模型管理 API ============
class ModelConfigRequest(BaseModel):
    enable_image: Optional[bool] = None
    enable_tracker: Optional[bool] = None
    enable_video: Optional[bool] = None

class ModelLoadRequest(BaseModel):
    model_type: str  # "image", "tracker", "video"

class ModelUnloadRequest(BaseModel):
    model_type: str  # "image", "tracker", "video"

@app.get("/models/status")
async def models_status():
    """获取当前模型加载状态和显存占用"""
    return {
        "success": True,
        **model_manager.get_model_status()
    }

@app.post("/models/configure")
async def models_configure(request: ModelConfigRequest):
    """配置要加载的模型(下次load_models时生效)"""
    if request.enable_image is not None:
        model_manager.enable_image = request.enable_image
    if request.enable_tracker is not None:
        model_manager.enable_tracker = request.enable_tracker
    if request.enable_video is not None:
        model_manager.enable_video = request.enable_video
    
    return {
        "success": True,
        "message": "配置已更新",
        "config": {
            "image": model_manager.enable_image,
            "tracker": model_manager.enable_tracker,
            "video": model_manager.enable_video
        }
    }

@app.post("/models/load")
async def models_load(request: ModelLoadRequest):
    """动态加载单个模型"""
    success = model_manager._load_single_model(request.model_type)
    if success:
        return {
            "success": True,
            "message": f"{request.model_type} 模型加载成功",
            **model_manager.get_model_status()
        }
    else:
        raise HTTPException(status_code=500, detail=f"{request.model_type} 模型加载失败")

@app.post("/models/unload")
async def models_unload(request: ModelUnloadRequest):
    """动态卸载单个模型释放显存"""
    success = model_manager._unload_single_model(request.model_type)
    if success:
        return {
            "success": True,
            "message": f"{request.model_type} 模型已卸载",
            **model_manager.get_model_status()
        }
    else:
        raise HTTPException(status_code=500, detail=f"{request.model_type} 模型卸载失败")

@app.post("/models/reload")
async def models_reload():
    """重新加载所有配置的模型（用于切换配置后）"""
    # 先卸载所有
    for model_type in ["image", "tracker", "video"]:
        model_manager._unload_single_model(model_type)
    
    # 再按配置加载
    success = model_manager.load_models()
    if success:
        return {
            "success": True,
            "message": "模型已重新加载",
            **model_manager.get_model_status()
        }
    else:
        raise HTTPException(status_code=500, detail="模型重新加载失败")

@app.post("/load_models")
async def load_models_endpoint(model_path: Optional[str] = None):
    """手动加载模型接口（兼容旧版）"""
    success = model_manager.load_models(model_path)
    if success:
        return {"success": True, "message": "SAM3 模型加载成功"}
    else:
        raise HTTPException(status_code=500, detail="模型加载失败")

# ============ 交互式分割 API ============
@app.post("/interactive/start", response_model=InteractiveStartResponse)
async def interactive_start(request: InteractiveStartRequest):
    """
    1. 开始交互式点击分割会话
    
    请求示例:
    {
        "image_base64": "..."
    }
    
    响应:
    {
        "success": true,
        "session_id": "uuid-string",
        "message": "交互式会话已创建"
    }
    """
    try:
        image = decode_image(request.image_base64)
        session_id = model_manager.start_interactive_session(image)
        
        return InteractiveStartResponse(
            success=True,
            session_id=session_id,
            message="交互式会话已创建"
        )
        
    except Exception as e:
        logger.error(f"创建交互式会话失败: {e}")
        import traceback
        traceback.print_exc()
        return InteractiveStartResponse(
            success=False,
            message=f"创建会话失败: {str(e)}"
        )

@app.post("/interactive/add_point", response_model=InteractiveAddPointResponse)
async def interactive_add_point(request: InteractiveAddPointRequest):
    """
    2. 添加点并执行分割
    
    请求示例:
    {
        "session_id": "uuid-string",
        "point": [500.0, 300.0],
        "label": 1
    }
    
    响应:
    {
        "success": true,
        "mask": [[...]],
        "shape": [H, W],
        "num_points": 3,
        "message": "点击分割完成"
    }
    """
    start_time = time.time()
    
    try:
        result = model_manager.add_point_and_predict(
            session_id=request.session_id,
            point=request.point,
            label=request.label
        )
        
        processing_time = (time.time() - start_time) * 1000
        
        return InteractiveAddPointResponse(
            success=True,
            mask=result["mask"],
            shape=result["shape"],
            num_points=result["num_points"],
            message=f"点击分割完成 ({processing_time:.1f}ms)"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"添加点分割错误: {e}")
        import traceback
        traceback.print_exc()
        return InteractiveAddPointResponse(
            success=False,
            message=f"分割失败: {str(e)}"
        )

@app.post("/interactive/pause")
async def interactive_pause(request: InteractivePauseRequest):
    """
    4. 暂停交互式点击分割会话
    
    请求示例:
    {
        "session_id": "uuid-string"
    }
    """
    try:
        result = model_manager.pause_interactive_session(request.session_id)
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"暂停会话错误: {e}")
        return {"success": False, "message": f"暂停失败: {str(e)}"}
    
@app.post("/interactive/resume")
async def interactive_resume(request:InteractiveResumeRequest):
    """
    恢复已暂停的交互式点击分割会话

    请求示例:
    {
        "session_id": "uuid-string"
    }
    """
    try:
        result = model_manager.resume_interactive_session(request.session_id)
        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"恢复会话错误: {e}")
        return {"success": False, "message": f"恢复失败: {str(e)}"}

    
@app.post("/interactive/clear")
async def interactive_clear(request: InteractiveClearRequest):
    """
    5. 清除用户提交的鼠标点击的点
    
    请求示例:
    {
        "session_id": "uuid-string"
    }
    """
    try:
        result = model_manager.clear_session_points(request.session_id)
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"清除点错误: {e}")
        return {"success": False, "message": f"清除失败: {str(e)}"}

@app.post("/interactive/save", response_model=InteractiveSaveResponse)
async def interactive_save(request: InteractiveSaveRequest):
    """
    6. 保存交互式分割结果
    
    请求示例:
    {
        "session_id": "uuid-string",
        "return_format": "mask"
    }
    
    响应:
    {
        "success": true,
        "mask": [[...]],
        "shape": [H, W],
        "message": "分割结果已保存"
    }
    """
    try:
        result = model_manager.save_session_result(
            session_id=request.session_id,
            return_format=request.return_format
        )
        
        return InteractiveSaveResponse(
            success=True,
            mask=result.get("mask"),
            mask_base64=result.get("mask_base64"),
            shape=result.get("shape"),
            message=result["message"]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"保存结果错误: {e}")
        import traceback
        traceback.print_exc()
        return InteractiveSaveResponse(
            success=False,
            message=f"保存失败: {str(e)}"
        )

@app.get("/interactive/sessions")
async def list_sessions():
    """列出所有活跃的交互式会话（调试用）"""
    sessions = []
    for session_id, session in model_manager.interactive_sessions.items():
        sessions.append({
            "session_id": session_id,
            "active": session["active"],
            "num_points": len(session["click_points"]),
            "has_result": session["last_mask"] is not None,
            "created_at": session["created_at"]
        })
    return {"sessions": sessions, "total": len(sessions)}

@app.delete("/interactive/session/{session_id}") # @app.delete装饰器说明, delete_session只会响应DELETE请求
async def delete_session(session_id: str):
    """删除指定会话"""
    if session_id in model_manager.interactive_sessions:
        del model_manager.interactive_sessions[session_id]
        return {"success": True, "message": f"会话 {session_id} 已删除"}
    else:
        raise HTTPException(status_code=404, detail=f"会话 {session_id} 不存在")

# ============ 点击分割接口 ============
@app.post("/predict/click", response_model=InferenceResponse)
async def predict_click_endpoint(request: ClickPromptRequest):
    """
    点击提示分割接口
    
    请求示例:
    {
        "image_base64": "...",
        "click_points": [[500.0, 300.0]],  # 像素坐标
        "click_labels": [1],                 # 1=正样本
        "width": 1000,
        "height": 652
    }
    """
    start_time = time.time()
    
    try:
        # 解码图片
        image = decode_image(request.image_base64)
        
        # 执行点击分割
        result = model_manager.predict_click(
            image=image,
            click_points=request.click_points,
            click_labels=request.click_labels,
            width=request.width,
            height=request.height
        )
        
        processing_time = (time.time() - start_time) * 1000
        
        return InferenceResponse(
            success=True,
            mask=result["mask"],
            shape=result["shape"],
            message=result["message"],
            processing_time_ms=processing_time
        )
        
    except Exception as e:
        logger.error(f"点击分割错误: {e}")
        import traceback
        traceback.print_exc()
        return InferenceResponse(
            success=False,
            message=f"点击分割失败: {str(e)}"
        )


# ============ 文本分割接口 ============
@app.post("/predict/text", response_model=InferenceResponse)
async def predict_text_endpoint(request: TextPromptRequest):
    """
    文本提示分割接口
    
    请求示例:
    {
        "image_base64": "...",
        "text_prompt": "person",
        "confidence_threshold": 0.5
    }
    """
    start_time = time.time()
    
    try:
        # 解码图片
        image = decode_image(request.image_base64)
        
        # 执行文本分割
        result = model_manager.predict_text(
            image=image,
            text_prompt=request.text_prompt,
            confidence_threshold=request.confidence_threshold
        )
        
        processing_time = (time.time() - start_time) * 1000
        
        return InferenceResponse(
            success=True,
            masks=result["masks"],
            scores=result["scores"],
            num_objects=result["num_objects"],
            message=result["message"],
            processing_time_ms=processing_time
        )
        
    except Exception as e:
        logger.error(f"文本分割错误: {e}")
        import traceback
        traceback.print_exc()
        return InferenceResponse(
            success=False,
            message=f"文本分割失败: {str(e)}"
        )


# ============ 框分割接口 ============
@app.post("/predict/box", response_model=InferenceResponse)
async def predict_box_endpoint(request: BoxPromptRequest):
    """
    框提示分割接口
    
    请求示例:
    {
        "image_base64": "...",
        "box_xywh": [480.0, 290.0, 110.0, 360.0]  # x, y, width, height
    }
    """
    start_time = time.time()
    
    try:
        # 解码图片
        image = decode_image(request.image_base64)
        
        # 执行框分割
        result = model_manager.predict_box(
            image=image,
            box_xywh=request.box_xywh
        )
        
        processing_time = (time.time() - start_time) * 1000
        
        return InferenceResponse(
            success=True,
            mask=result["mask"],
            shape=result["shape"],
            message=result["message"],
            processing_time_ms=processing_time
        )
        
    except Exception as e:
        logger.error(f"框分割错误: {e}")
        import traceback
        traceback.print_exc()
        return InferenceResponse(
            success=False,
            message=f"框分割失败: {str(e)}"
        )


# ============ 文件上传接口（备用） ============
# 可以传点击point的文件, 用JSON字符串来传
@app.post("/predict/click_file")
async def predict_click_file_endpoint(
    file: UploadFile = File(...),
    click_points: str = "[]",      # JSON 字符串 [[x, y], ...]
    click_labels: str = "[]",      # JSON 字符串 [1, 0, ...]
    width: int = 0,
    height: int = 0
):
    """点击分割 - 文件上传方式"""
    import json
    start_time = time.time()
    
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        points = json.loads(click_points)
        labels = json.loads(click_labels)
        
        if width == 0 or height == 0:
            width, height = image.size
        
        result = model_manager.predict_click(
            image=image,
            click_points=[(p[0], p[1]) for p in points],
            click_labels=labels,
            width=width,
            height=height
        )
        
        processing_time = (time.time() - start_time) * 1000
        
        return {
            "success": True,
            "mask": result["mask"],
            "shape": result["shape"],
            "message": result["message"],
            "processing_time_ms": processing_time
        }
        
    except Exception as e:
        logger.error(f"文件上传分割错误: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============ 启动 ============
if __name__ == "__main__":
    # 从环境变量读取配置，默认 8000
    import argparse
    
    parser = argparse.ArgumentParser(description="SAM3 推理服务")
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", 8000)), help="服务端口 (默认: 8000)")
    parser.add_argument("--host", type=str, default=os.environ.get("HOST", "0.0.0.0"), help="绑定地址 (默认: 0.0.0.0)")
    parser.add_argument("--model-path", type=str, default="/root/workspace/modelRepo/SAM3", help="模型路径")
    args = parser.parse_args()
    
    # 启动时自动加载模型
    model_manager.model_path = args.model_path
    model_manager.load_models()
    
    logger.info(f"🚀 启动 SAM3 推理服务: http://{args.host}:{args.port}")
    logger.info(f"📋 健康检查: GET http://{args.host}:{args.port}/health")
    logger.info(f"👆 单次点击分割: POST http://{args.host}:{args.port}/predict/click")
    logger.info(f"📝 文本分割: POST http://{args.host}:{args.port}/predict/text")
    logger.info(f"📦 框分割:   POST http://{args.host}:{args.port}/predict/box")
    logger.info(f"🎯 交互式分割开始: POST http://{args.host}:{args.port}/interactive/start")
    logger.info(f"➕ 交互式添加点:   POST http://{args.host}:{args.port}/interactive/add_point")
    logger.info(f"⏸️ 交互式暂停:     POST http://{args.host}:{args.port}/interactive/pause")
    logger.info(f"🧹 交互式清除点:   POST http://{args.host}:{args.port}/interactive/clear")
    logger.info(f"💾 交互式保存:     POST http://{args.host}:{args.port}/interactive/save")
    logger.info(f"⚙️  模型管理: GET  http://{args.host}:{args.port}/models/status")
    
    uvicorn.run(app, host=args.host, port=args.port)

"""图像分割业务服务: 会话/组操作/推理/提示文件"""

import torch
from PIL import Image
from typing import Dict, List, Optional

from .models import ImageSession
from .session_manager import SessionManager


class ImageService:
    """图像域业务逻辑(构造注入会话管理器与计算引擎)"""

    def __init__(self, session_manager: SessionManager, compute_engine):
        self.session_manager = session_manager
        self.compute_engine = compute_engine

    # ========== 会话 ==========
    def create_image_session(self, image: Image.Image) -> str:
        return self.session_manager.register_image_session(image)

    def close_image_session(self, session_id: str) -> Dict:
        ok = self.session_manager.delete_image_session(session_id)
        if not ok:
            raise ValueError(f"会话 {session_id} 不存在")
        return {"success": True, "message": f"会话 {session_id} 已删除"}

    # ========== 组操作 ==========
    def add_point_to_group(self, session_id: str, group_id: int,
                           x: float, y: float, label: int) -> Dict:
        session = self.session_manager.get_image_session(session_id)
        if session is None:
            raise ValueError(f"会话 {session_id} 不存在")
        group = session.get_or_create_group(group_id)
        group.add_point(x, y, label)
        return {
            "success": True,
            "group_id": group_id,
            "num_points": len(group.points),
            "message": "点已添加，调用 predict 进行推理",
        }

    def add_box_to_group(self, session_id: str, group_id: int,
                         x1: float, y1: float, x2: float, y2: float) -> Dict:
        session = self.session_manager.get_image_session(session_id)
        if session is None:
            raise ValueError(f"会话 {session_id} 不存在")
        group = session.get_or_create_group(group_id)
        group.set_box(x1, y1, x2, y2)
        return {
            "success": True,
            "group_id": group_id,
            "has_box": True,
            "message": "框已添加，调用 predict 进行推理",
        }

    # 对于图片的清除, 在服务层直接实现, 因为图片没有会话状态
    # 清理点和框, 但不删除组
    def clear_group(self, session_id: str, group_id: int) -> Dict:
        session = self.session_manager.get_image_session(session_id)
        if session is None:
            raise ValueError(f"会话 {session_id} 不存在")
        if group_id in session.point_groups:
            # 将点组clear, 我印象里推力refine, 就是要重传,
            # refine只是把image_embedding存起来了,来节约时间
            # (节约在Encoder的计算, 具体可以节约多少没实际测过)
            session.point_groups[group_id].clear()
        return {
            "success": True,
            "group_id": group_id,
            "message": "组已清空，调用 predict 进行推理",
        }

    def delete_image_point(self, session_id: str, group_id: int,
                           point_index: int = -1) -> Dict:
        """
        删除图像某组中的单个点(默认最后一个, 支持撤销式交互)
        图像模型无状态, 下次 predict 自然用剩余点重算, 无需额外通知
        """
        session = self.session_manager.get_image_session(session_id)
        if session is None:
            raise ValueError(f"会话 {session_id} 不存在")
        group = session.point_groups.get(group_id)
        if group is None or not group.points:
            raise ValueError(f"组 {group_id} 没有可删除的点")
        if not -len(group.points) <= point_index < len(group.points):
            # 这还能倒着删, 有点抽象了
            raise ValueError(f"point_index 越界: {point_index}, 共 {len(group.points)} 个点")
        del group.points[point_index]
        del group.labels[point_index]
        return {
            "success": True,
            "group_id": group_id,
            "num_points": len(group.points),
            "message": "点已删除，调用 predict 进行推理",
        }

    # 一个组只有一个框
    def clear_image_box(self, session_id: str, group_id: int) -> Dict:
        """清除图像某组中的框提示(点保留), 下次 predict 生效"""
        session = self.session_manager.get_image_session(session_id)
        if session is None:
            raise ValueError(f"会话 {session_id} 不存在")
        group = session.point_groups.get(group_id)
        if group is None or group.box is None:
            raise ValueError(f"组 {group_id} 没有框")
        group.box = None
        return {
            "success": True,
            "group_id": group_id,
            "has_box": False,
            "message": "框已清除，调用 predict 进行推理",
        }

    # 删除组
    def delete_group(self, session_id: str, group_id: int) -> Dict:
        session = self.session_manager.get_image_session(session_id)
        if session is None:
            raise ValueError(f"会话 {session_id} 不存在")
        if group_id in session.point_groups:
            del session.point_groups[group_id]

        # 从 masks 中删除对应 group, 这段写的有点啰嗦
        if session.masks is not None and group_id in session.group_ids:
            idx = session.group_ids.index(group_id)
            mask_list = [session.masks[i] for i in range(session.masks.shape[0]) if i != idx]
            session.group_ids.pop(idx)
            session.masks = torch.stack(mask_list, dim=0) if mask_list else None

        # 如果所有组都删完了，清除 embeddings
        if not session.point_groups:
            session.image_embeddings = None
            session.masks = None
            session.group_ids = []

        return {
            "masks_tensor": session.masks,
            "num_objects": session.masks.shape[0] if session.masks is not None else 0,
            "group_ids": session.group_ids,
        }

    # ========== 推理 ==========
    def _predict_image_session(self, session: ImageSession) -> Dict:
        """
        执行图像推理(核心方法)

        按类型分组推理，合并结果按 group_id 排序
        """
        pure_point, pure_box, mixed = session.classify_groups()

        if not pure_point and not pure_box and not mixed:
            raise ValueError("没有可用的提示")

        # 记录各类型 group_id 和对应的 mask
        type_results = []  # [(group_ids, masks_tensor), ...]

        # 纯点推理
        if pure_point:
            pts, lbls, _ = session._groups_to_tensor(pure_point)
            group_ids = [g.group_id for g in pure_point]

            if session.image_embeddings is None:
                # 第一次算结果
                result = self.compute_engine.predict_prompt(
                    image=session.image,
                    click_points=pts,
                    click_labels=lbls,
                )
                session.image_embeddings = result["image_embeddings"]
            else:
                # 对于Sam3TrackerModel是支持输入mask去refine的,
                # 但是对于Sam3TrackerVideoModel,当你输入mask的时候, 会直接将你输入的mask作为结果
                prev_mask = self._extract_masks(session, group_ids)
                result = self.compute_engine.predict_prompt(
                    image=None,
                    click_points=pts,
                    click_labels=lbls,
                    input_masks=prev_mask,
                    image_embeddings=session.image_embeddings,
                    original_size=(session.image.size[1], session.image.size[0]),
                )

            # 将纯点组的group的结果存在results里
            type_results.append((group_ids, result["masks"]))

        # 纯框推理
        if pure_box:
            _, _, boxes = session._groups_to_tensor(pure_box) # 只需要boxes
            group_ids = [g.group_id for g in pure_box]

            if session.image_embeddings is None:
                result = self.compute_engine.predict_prompt(
                    image=session.image,
                    input_boxes=boxes,
                )
                session.image_embeddings = result["image_embeddings"]
            else:
                prev_mask = self._extract_masks(session, group_ids)
                result = self.compute_engine.predict_prompt(
                    image=None,
                    input_boxes=boxes,
                    input_masks=prev_mask,
                    image_embeddings=session.image_embeddings,
                    original_size=(session.image.size[1], session.image.size[0]),
                )

            # 将纯框组的group的结果存在results里
            type_results.append((group_ids, result["masks"]))

        # 混合推理
        if mixed:
            pts, lbls, boxes = session._groups_to_tensor(mixed)
            group_ids = [g.group_id for g in mixed]

            if session.image_embeddings is None:
                result = self.compute_engine.predict_prompt(
                    image=session.image,
                    click_points=pts,
                    click_labels=lbls,
                    input_boxes=boxes,
                )
                session.image_embeddings = result["image_embeddings"]
            else:
                prev_mask = self._extract_masks(session, group_ids)
                result = self.compute_engine.predict_prompt(
                    image=None,
                    click_points=pts,
                    click_labels=lbls,
                    input_boxes=boxes,
                    input_masks=prev_mask,
                    image_embeddings=session.image_embeddings,
                    original_size=(session.image.size[1], session.image.size[0]),
                )

            # 将点框结合的group的结果存在results里
            type_results.append((group_ids, result["masks"]))

        # 按照group_id排序结果
        # type_results的的shape是
        # [
        #   (point_group_ids,point_results["masks"]),
        #   (box_group_ids,box_results["masks"]),
        #   (mix_group_ids, mix_results["masks"]),
        # ]
        all_items = []
        for group_ids, masks in type_results:
            for gid, mask in zip(group_ids, masks):
                all_items.append((gid, mask))

        all_items.sort(key=lambda x: x[0]) # all_items的shape是[(group_id, mask),.....]

        if all_items:
            session.group_ids = [gid for gid, _ in all_items]
            session.masks = torch.stack([mask for _, mask in all_items], dim=0)
        else:
            session.group_ids = []
            session.masks = None

        return {
            "masks_tensor": session.masks,
            "num_objects": session.masks.shape[0] if session.masks is not None else 0,
            "group_ids": session.group_ids,
        }

    def _extract_masks(self, session: ImageSession, group_ids: List[int]) -> Optional[torch.Tensor]:
        """从 session 中提取指定 group_ids 的 mask"""
        if session.masks is None or not session.group_ids:
            return None

        indices = []
        for gid in group_ids:
            # 这些gid有些有可能是非法的
            try:
                idx = session.group_ids.index(gid)
                indices.append(idx)
            except ValueError:
                # 即便没有也会return None
                return None

        return session.masks[indices]

    # 今天没弄完, 以下是明天的TODO
    # 需要把image_prompt和video_prompt文件格式的示例给到assets里面 done,
    # 并且要把image_prompt和video_prompt的定义再api.md里说明白 done
    # 需要把load_prompt按照Kimi Code里给的代码改为load_image_prompt_file和load_video_prompt_file done
    # 补上ImagePrompt和VideoPromptFile的数据类型 done
    # 审查完最后的predit_image 经过设查后,将add_prompt与_predict_image解耦
    def load_image_prompt_file(self, session_id: str, file_data: Dict,
                                merge_mode: str = "append") -> Dict:
        """
        加载图像 prompt 文件

        merge_mode:
        - "append": 追加到现有组（点追加，框覆盖）
        - "replace": 覆盖整个组（清空后重新加载）
        - "skip": 跳过已存在的组
        """
        session = self.session_manager.get_image_session(session_id)
        if session is None:
            raise ValueError(f"会话 {session_id} 不存在")
        if merge_mode not in ("append", "replace", "skip"):
            raise ValueError(f"merge_mode 必须是 append/replace/skip 之一，当前: {merge_mode}")

        if file_data.get("type") != "image":
            raise ValueError(f"期望 type='image'，实际为 '{file_data.get('type')}'")

        loaded_groups = []
        skipped_groups = []

        for group_data in file_data.get("groups", []):
            group_id = group_data["group_id"]

            # skip 模式：已存在则跳过
            if merge_mode == "skip" and group_id in session.point_groups:
                skipped_groups.append(group_id)
                continue

            # replace 模式：清空现有组
            if merge_mode == "replace" and group_id in session.point_groups:
                session.point_groups[group_id].clear()

            group = session.get_or_create_group(group_id)

            points = group_data.get("points", [])
            labels = group_data.get("labels", [])

            if len(points) != len(labels):
                raise ValueError(f"组 {group_id}: points({len(points)}) 和 labels({len(labels)}) 长度不匹配")

            for pt, lbl in zip(points, labels):
                group.add_point(pt[0], pt[1], lbl)

            box = group_data.get("box")
            if box is not None:
                group.set_box(box[0], box[1], box[2], box[3])

            loaded_groups.append({
                "group_id": group_id,
                "num_points": len(points),
                "has_box": box is not None,
            })

        return {
            "success": True,
            "session_id": session_id,
            "merge_mode": merge_mode,
            "groups_loaded": loaded_groups,
            "groups_skipped": skipped_groups,
            "total_groups": len(session.point_groups),
            "message": f"Prompt 已加载（模式: {merge_mode}），调用 predict 进行推理",
        }

    def predict_image(self, session_id: str) -> Dict:
        """
        触发图像推理(用于交互式会话)
        屏蔽掉细节, 供路由层调用
        在 load_prompt 或添加点/框后手动触发推理
        """
        session = self.session_manager.get_image_session(session_id)
        if session is None:
            raise ValueError(f"会话 {session_id} 不存在")
        return self._predict_image_session(session)

    def predict_image_once(self, image: Image.Image, groups: List[Dict]) -> Dict:
        """
        单次图像分割(无会话，直接推理)

        注意：此功能可能不够原子化，是否保留看后续整体使用。
        这个方法可能会取代 _predict_image, 或者被 _predict_image 取代。
        因为 _predict_image 现在也改为主动触发才会去分割。
        但这个函数比较偏先把提示点完，没有会话了，直接结束。
        """
        session = ImageSession(session_id="temp", image=image)
        for i, group_data in enumerate(groups):
            group = session.get_or_create_group(i)
            for pt, lbl in zip(group_data.get("points", []), group_data.get("labels", [])):
                group.add_point(pt[0], pt[1], lbl)
            if "box" in group_data:
                b = group_data["box"]
                group.set_box(b[0], b[1], b[2], b[3])
        return self._predict_image_session(session)

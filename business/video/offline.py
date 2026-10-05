"""
离线视频会话服务: 文件路径建会话 / 提示文件导入 / 结果查询

传播(submit_video_prompts)与会话创建(create_video_session)在 base.py
(双场景方法, 内部按 is_streaming 分支), 此处只留纯离线专属逻辑
"""

from typing import Dict

from PIL import Image

from .base import VideoSessionServiceBase


class OfflineVideoService(VideoSessionServiceBase):
    """离线视频业务逻辑"""

    # ========== 会话创建(纯离线: 视频文件) ==========
    def create_video_session_from_path(self, video_path, frame_stride=1,
                                   auto_predict=False, chunk_size=32) -> str:
        """离线长视频: decord 按段解码"""
        import decord # 这个库是Amazon开发维护的, 专门用来给深度学习用的视频解码库, 原生支持GPU解码, 以及Pytorch,
        # 不用像opencv一样手动搞很多步骤
        vr = decord.VideoReader(video_path) # vr is short for VideoReader
        num = len(vr)
        chunks = ([Image.fromarray(f) for f in
               vr.get_batch(list(range(s, min(s + chunk_size, num)))).asnumpy()] # []内的已经是第二层循环了,得到的是chunk, ([Image1, Image2, ...])
              for s in range(0, num, chunk_size))
        return self._create_from_chunks(chunks, num, frame_stride, auto_predict)

    # ========== 提示文件导入 ==========
    def load_video_prompt_file(self, session_id: str, file_data: Dict,
                               merge_mode: str = "append") -> Dict:
        """
        加载视频 prompt 文件（批量导入多帧提示）

        merge_mode:
        这个业务逻辑和图像是一样的, 只是在不同的帧去做
        - "append": 追加到现有（同同组存在则叠加）
        - "replace": 覆盖同帧同组（清空该帧该组后重新加载）
        - "skip": 跳过已存在的同帧同组
        """
        session = self._get_video_session(session_id)
        if merge_mode not in ("append", "replace", "skip"):
            raise ValueError(f"merge_mode 必须是 append/replace/skip 之一，当前: {merge_mode}")

        if file_data.get("type") != "video":
            raise ValueError(f"期望 type='video'，实际为 '{file_data.get('type')}'")

        loaded_frames = set() # 用于记录哪些帧被导入了提示
        skipped_groups = [] # 记录哪些(帧号, 组号)因为merge_mode='skip'被跳过

        for frame_data in file_data.get("frames", []):
            frame_idx = frame_data["frame_idx"]
            frame_loaded = False    # 该帧是否有至少一个组真正被加载(全被 skip 则不算)

            for group_data in frame_data.get("groups", []):
                group_id = group_data["group_id"]

                existing = session.frame_prompts.get(frame_idx, {}).get(group_id) # 找出已经存在的组(这个是在服务层的,不是计算层)

                # skip 模式：已存在则跳过
                if merge_mode == "skip" and existing is not None:
                    skipped_groups.append({"frame_idx": frame_idx, "group_id": group_id})
                    continue

                # replace 模式：清空该帧该组(服务层清空 + 已提交过则同步清底层:
                # 将这组的输入与输出彻底删除, 即便这一帧的组变为了空, 也能解决)
                # 这个也比较危险吧, 直接用了计算层的操作
                if merge_mode == "replace" and existing is not None:
                    existing.clear()
                    obj_id = self._obj_id_of(session, group_id)
                    if obj_id is not None:
                        self.compute_engine.remove_video_object_inputs(
                            session.video_session, obj_id,
                            self._to_session_idx(session, frame_idx))

                group = self._record_prompt(session, group_id, frame_idx) # 这个是用来给服务层做记账的函数, 用来得到服务层已经记录过的group

                points = group_data.get("points", [])
                labels = group_data.get("labels", [])
                if len(points) != len(labels):
                    raise ValueError(
                        f"帧 {frame_idx} 组 {group_id}: points({len(points)}) 和 "
                        f"labels({len(labels)}) 长度不匹配")

                for (x, y), lbl in zip(points, labels):
                    group.add_point(float(x), float(y), int(lbl))

                box = group_data.get("box")
                if box is not None:
                    group.set_box(float(box[0]), float(box[1]),
                                  float(box[2]), float(box[3]))

                frame_loaded = True

            if frame_loaded:
                loaded_frames.add(frame_idx)

        # 提示批量变化后, 最早导入帧及之后的缓存作废(前向因果)
        if loaded_frames:
            self._invalidate_from(session, min(loaded_frames))

        return {
            "success": True,
            "session_id": session_id,
            "merge_mode": merge_mode,
            "loaded_frames": sorted(loaded_frames),
            "skipped_groups": skipped_groups,
            "total_prompt_frames": len(session.frame_prompts),
            "message": "Prompt 已加载，调用 submit_video_prompts 进行传播",
        }

    # ========== 结果查询 ==========
    def get_video_frame_result(self, session_id: str, frame_idx: int,
                               compute_if_missing: bool = False) -> Dict:
        """
        取某帧结果: 已算直接返回; 非关键帧复用最近前序关键帧
        compute_if_missing=True 且超出跟踪前沿时先补算到该帧(边看边分割的情况下使用)
        """
        session = self._get_video_session(session_id)
        if compute_if_missing and frame_idx not in session.keyframe_results:
            frontier = session.last_computed_frame # 最新算完的帧
            if frontier is None or frame_idx > frontier:
                # 如果超出了最新的前序关键帧, 补算到该帧
                for _ in self.submit_video_prompts(session_id, end_frame=frame_idx):
                    pass
        res = session.keyframe_results.get(frame_idx) # 查这帧有没有算过, 算过直接返回
        if res is not None:
            return {"frame_idx": frame_idx, "keyframe": True,
                    **self._visible_result(session, frame_idx, res)}
        prev = self._latest_cached_before(session, frame_idx) # 不然的话用关键帧顶替
        if prev is None:
            raise ValueError(f"第 {frame_idx} 帧之前没有已计算的关键帧, 请先 submit")
        return {"frame_idx": frame_idx, "keyframe": False, "reused_from": prev,
                **self._visible_result(session, frame_idx, session.keyframe_results[prev])}

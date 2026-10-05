"""
视频会话服务基类: 离线/流式共享的辅助逻辑 + 提示操作 + 生命周期

场景差异全部在子类(offline.py: 传播计划; streaming.py: 推帧/网格/插帧),
本基类只放两个场景共用的部分
"""

import torch
from typing import Dict, Generator, Optional, Set, Tuple

from ..models import VideoSession, PointGroup
from ..session_manager import SessionManager


class VideoSessionServiceBase:
    """视频会话共享逻辑(构造注入会话管理器与计算引擎)"""

    def __init__(self, session_manager: SessionManager, compute_engine):
        self.session_manager = session_manager
        self.compute_engine = compute_engine

    # ========== 会话获取 ==========
    def _get_video_session(self, session_id: str) -> VideoSession:
        session = self.session_manager.get_video_session(session_id)
        if session is None:
            raise ValueError(f"视频会话 {session_id} 不存在")
        return session

    # ========== group_id 与底层 obj_id 的映射(服务层管 id, 计算层不管) ==========
    # 类方法, 不依赖实例状态, 将前端和服务层的对象编号映射到计算层的group_id
    @staticmethod
    def _obj_id_of(session: VideoSession, group_id: int) -> Optional[int]:
        """submitted_groups[obj_id] = group_id; 未提交过返回 None"""
        try:
            return session.submitted_groups.index(group_id)
        except ValueError:
            return None

    # obj_id 不是加提示时就分配，而是提交计算时才分配,
    # 也就是第一次把某组的提示真正推给底层的前一刻才分配
    def _ensure_obj_id(self, session: VideoSession, group_id: int) -> int:
        obj_id = self._obj_id_of(session, group_id)
        if obj_id is None:
            session.submitted_groups.append(group_id)
            obj_id = len(session.submitted_groups) - 1
        return obj_id

    # 用于映射帧号
    @staticmethod
    def _to_session_idx(session: VideoSession, frame_idx: int) -> int:
        """客户端帧号 -> 底层 session 帧号(离线恒等, 流式查映射)"""
        if session.is_streaming:
            return session.stream_c2s[frame_idx]
        return frame_idx

    # ---- 提示提交与计算 ----

    def _flush_dirty_frame(self, session: VideoSession, frame_idx: int) ->None:
        """
        把某帧的脏提示推入计算层, 当往某帧加过提示,
        之后再加提示, 是将结果重算, 而不是只算增量
        """
        for (f, gid) in [d for d in session.dirty_prompts if d[0] == frame_idx]: # d[0]为frame_idx, 并且一个帧可能对应多个组
            group = session.frame_prompts.get(f, {}).get(gid)
            session.dirty_prompts.discard((f, gid)) # 字典的删除方法, 如果不存在, 什么也不做, 不报错

            # 没有组 or 组内没有提示
            if group is None or (not group.points and group.box is None):
               continue

            obj_id = self._ensure_obj_id(session, gid) # 为group设置obj_id
            pts = [list(p) for p in group.points] if group.points else None
            # 流式会话 init 时无帧, 字典里是 (None, None), 会把底层
            # video_height 覆盖成 None; original_size 在首帧推入时记录
            self.compute_engine.add_video_prompt(
                session.video_session,
                frame_idx=self._to_session_idx(session, f),
                obj_id=obj_id,
                click_points=[[pts]] if pts else None,
                click_labels=[[group.labels]] if pts else None,
                input_boxes=[[list(group.box)]] if group.box is not None else None,
                original_size=session.original_size,
            ) # 补丁4a, 为了解决流式会话的问题, 但这个应该要路由层来承担这个责任, 给original_size


    def _cache_result(self, session: VideoSession, frame_idx: int,
                      masks: Optional[torch.Tensor]) -> Dict:
        """缓存一帧的计算结果(掩码搬回 CPU, groups 记录行对齐)"""
        if session.anchor_frame is None:
            session.anchor_frame = frame_idx
        res = {
            "masks": masks.cpu() if masks is not None else None,
            "groups": list(session.submitted_groups),
        }
        session.keyframe_results[frame_idx] = res # 将keyframe的结果缓存
        session.last_computed_frame = (frame_idx if session.last_computed_frame is None
                                       else max(session.last_computed_frame, frame_idx)) # 如果帧为比当前帧靠后, 则将其作为最新的一帧

        return res

    def _compute_and_cache(self, session: VideoSession, frame_idx: int) -> Dict:
        """计算某个客户端帧并缓存(0 物体防护: 没物体不调用底层)"""
        if not session.submitted_groups:
            return self._cache_result(session, frame_idx, None) # 0物体防护, 不会调用底层的engine
        out = self.compute_engine.predict_video_frame(
            session.video_session, self._to_session_idx(session, frame_idx))
        return self._cache_result(session, frame_idx, out["masks"])

    def _visible_result(self, session: VideoSession, frame_idx: int, res: Dict) -> Dict:
        """过滤掉在物体第一次提示之前的帧出现的mask, 只有第一次提示之后的mask才能看到"""
        keep = [i for i, g in enumerate(res["groups"])
                if session.group_first_frame.get(g, 0) <= frame_idx] # frame_idx在group_first_frame之后的,保留
        masks = res["masks"]
        if masks is not None:
            masks = masks[keep] if keep else None
        return {"masks": masks, "groups": [res["groups"][i] for i in keep]}

    def _latest_cached_before(self, session: VideoSession, frame_idx: int) -> Optional[int]:
        # 找一个不大于frame_idx的, 最近的一个已缓存的关键帧
        # 用于插帧复用之前的掩码
        candidates = [f for f in session.keyframe_results if f <= frame_idx]
        return max(candidates) if candidates else None

    def _invalidate_from(self, session: VideoSession, frame_idx: int) -> None:
        """提示变化后丢弃 frame_idx 及之后的缓存(前向因果, 之前的仍有效)"""
        for f in [f for f in session.keyframe_results if f >= frame_idx]:
            del session.keyframe_results[f] # 当我删除idx的帧的时候, 我将idx这帧和之后处理过的帧的结果全部删除
        session.last_computed_frame = max(session.keyframe_results) if session.keyframe_results else None
        if session.last_computed_frame is None:
            session.anchor_frame = None

    def _remove_group_everywhere(self, session: VideoSession, group_id: int) -> None:
        """移除一个组的对应的物体的全部痕迹: 底层物体、提示记录、缓存中的对应行"""
        obj_id = self._obj_id_of(session, group_id)
        if obj_id is not None:
            self.compute_engine.remove_video_object(session.video_session, obj_id) # 底层删除掉这个物体
            session.submitted_groups.pop(obj_id)  # 底层已重索引, 列表同步前移
        for f in list(session.frame_prompts):
            session.frame_prompts[f].pop(group_id, None)
            if not session.frame_prompts[f]:
                del session.frame_prompts[f]
        session.dirty_prompts = {d for d in session.dirty_prompts if d[1] != group_id} # 将未提交的group_id的物体的提示也删除
        session.group_first_frame.pop(group_id, None)
        # 缓存掩码的行与 obj_id 对齐, 删对应行即可(其他物体的追踪互不影响)
        for res in session.keyframe_results.values():
            if group_id in res["groups"]:
                idx = res["groups"].index(group_id) # 得到对应的obj_idx
                keep = [i for i in range(len(res["groups"])) if i != idx] # 要保留的obj
                res["masks"] = res["masks"][keep] if (res["masks"] is not None and keep) else None
                res["groups"] = [res["groups"][i] for i in keep]

    # ---- 加提示(记录与计算解耦) ----
    def _record_prompt(self, session: VideoSession, group_id: int, frame_idx: int) -> PointGroup:
        group = session.frame_prompts.setdefault(frame_idx, {}).get(group_id)
        if group is None:
            # group is None会出现在这帧上, 这个group第一次出现
            group = PointGroup(group_id=group_id)
            session.frame_prompts[frame_idx][group_id] = group
        session.group_first_frame[group_id] = min(
            session.group_first_frame.get(group_id, frame_idx), frame_idx) # 是否将该帧作为这个group的第一个关键帧
        session.dirty_prompts.add((frame_idx, group_id))
        return group

    @staticmethod
    def _check_stream_prompt_frame(session: VideoSession, frame_idx: int) -> None:
        if session.is_streaming and frame_idx != session.received_frame_count - 1:
            raise ValueError("流式模式只能给最新收到的帧加提示") # 检查该帧是否为最新一帧


    def _ensure_prompt_frame_pushed(self, session: VideoSession, frame_idx: int) -> None:
        """
        流式下提示帧若尚未推入底层(非网格帧), 先补推——提示帧必成关键帧
        这个函数本身的设计就不是给非网格帧用的
        危险函数(待后续优化), 已知三个问题:
        2. 不均匀时间步: 补推使底层帧序列间隔不等(如 0,3,6,7), 记忆注意力的时间
           位置编码信号变脏, 运动剧烈时可能影响精度
        3. 并发窗口: 依赖 last_frame 就是 frame_idx 那一帧, 若路由层并发处理
           推帧与提示消息可能推错帧(顺序处理则无此问题)
        """
        if not session.is_streaming or frame_idx in session.stream_c2s:
            return  # 离线无需补推; 已推入底层的帧直接返回, 如果在session.stream_c2s, 则意味着该帧已经被推入底层
        if frame_idx != session.received_frame_count - 1:
            # 契约: 只能补推最新收到的帧(底层只能顺序接帧)
            raise ValueError(
                f"只能补推最新帧: frame_idx={frame_idx}, "
                f"最新={session.received_frame_count - 1}")
        self._push_streaming_frame(session, session.last_frame, track=False) # 补推只注册帧，之后照旧 _flush_dirty_frame → _compute_and_cache，只追踪一次

    # ========== 提示操作(离线/流式共用) ==========
    def add_video_point(self, session_id: str, group_id: int,
                        x: float, y: float, label: int, frame_idx: int) -> Dict:
        session = self._get_video_session(session_id)
        self._check_stream_prompt_frame(session, frame_idx)
        group = self._record_prompt(session, group_id, frame_idx)
        group.add_point(x, y, label)
        if not session.auto_predict:
            # 该帧及之后的旧缓存不含新提示, 作废
            # (否则前沿不退, 增量 submit 跳过这段, 提示永远算不上)
            # 是需要回滚前沿，重新propagate的
            self._invalidate_from(session, frame_idx) # 标记前面的掩码缓存脏了
            return {"success": True, "computed": False,
                    "message": "提示已记录, 调用 submit 计算"}
        self._ensure_prompt_frame_pushed(session, frame_idx)
        self._flush_dirty_frame(session, frame_idx) # 将prompt传给计算层
        res = self._compute_and_cache(session, frame_idx) # 返回结果
        # 该帧已用新提示重算(所以不杀本帧), 之后帧的旧缓存作废, 前沿回退到该帧
        self._invalidate_from(session, frame_idx + 1)
        return {"success": True, "computed": True, "frame_idx": frame_idx,
                **self._visible_result(session, frame_idx, res)}

    def add_video_box(self, session_id: str, group_id: int,
                    x1: float, y1: float, x2: float, y2: float, frame_idx: int) -> Dict:
        session = self._get_video_session(session_id)
        self._check_stream_prompt_frame(session, frame_idx)
        group = self._record_prompt(session, group_id, frame_idx)
        group.set_box(x1, y1, x2, y2)
        if not session.auto_predict:
            return {"success": True, "computed": False,
                    "message": "提示已记录, 调用 submit 计算"}
        self._ensure_prompt_frame_pushed(session, frame_idx)
        self._flush_dirty_frame(session, frame_idx)
        self._invalidate_from(session, frame_idx) # 标记前面的掩码缓存脏了
        res = self._compute_and_cache(session, frame_idx)
        # 该帧已用新提示重算(所以不杀本帧), 之后帧的旧缓存作废, 前沿回退到该帧
        self._invalidate_from(session, frame_idx + 1)
        return {"success": True, "computed": True, "frame_idx": frame_idx,
                **self._visible_result(session, frame_idx, res)}

    # ---- 删提示(细粒度: 单点/框; 空组自动升级为删除物体) ----
    def delete_video_point(self, session_id: str, group_id: int,
                           frame_idx: int, point_index: int) -> Dict:
        session = self._get_video_session(session_id)
        group = session.frame_prompts.get(frame_idx, {}).get(group_id)
        if group is None or point_index >= len(group.points):
            raise ValueError("该点不存在")
        if not -len(group.points) <= point_index < len(group.points): # 和image的delete_video_point一样
            raise ValueError(f"point_index 越界: {point_index}, 共 {len(group.points)} 个点")
        group.points.pop(point_index)
        group.labels.pop(point_index)
        self._after_prompt_fine_removed(session, group_id, frame_idx)
        return {"success": True, "remaining_points": len(group.points)}

    def clear_video_box(self, session_id: str, group_id: int, frame_idx: int) -> Dict:
        session = self._get_video_session(session_id)
        group = session.frame_prompts.get(frame_idx, {}).get(group_id)
        if group is None or group.box is None:
            raise ValueError("该框不存在")
        group.box = None
        self._after_prompt_fine_removed(session, group_id, frame_idx)
        return {"success": True}

    def _after_prompt_fine_removed(self, session: VideoSession, group_id: int, frame_idx: int) -> None:
        """细粒度删除善后: 同步底层输入 -> 重算最早提示帧 -> 失效缓存/空组升级删除"""
        group = session.frame_prompts.get(frame_idx, {}).get(group_id)
        obj_id = self._obj_id_of(session, group_id)
        if group is not None and (group.points or group.box is not None):
            if obj_id is not None:
                session.dirty_prompts.add((frame_idx, group_id))  # 剩余提示重推覆盖
        else:
            # 该帧提示已空: 清掉底层该帧的输入和输出
            if obj_id is not None:
                # 如果对象还存在
                self.compute_engine.remove_video_object_inputs(
                    session.video_session, obj_id, self._to_session_idx(session, frame_idx)) # 清除掉底层该帧的输入
            session.dirty_prompts.discard((frame_idx, group_id)) # 将未提交的提示删除
            if frame_idx in session.frame_prompts:
                session.frame_prompts[frame_idx].pop(group_id, None) # 删除掉该帧上的空组
                if not session.frame_prompts[frame_idx]:
                    del session.frame_prompts[frame_idx]  # 如果该帧的上的所有组都删光了

        # 该组在所有帧上都没有提示了 -> 空组升级, 删除整个物体
        if not any(group_id in groups for groups in session.frame_prompts.values()):
            self._remove_group_everywhere(session, group_id)
        # 只有提示到达过底层, 缓存才可能因此变脏: frame_idx 及之后作废(前向因果)
        if obj_id is not None:
            self._invalidate_from(session, frame_idx)

    def clear_video_group(self, session_id: str, group_id: int) -> Dict:
        """删除一个物体(组)及其在全部帧上的提示"""
        session = self._get_video_session(session_id)
        self._remove_group_everywhere(session, group_id)
        remaining = sorted({g for groups in session.frame_prompts.values() for g in groups})
        return {"success": True, "group_id": group_id, "remaining_groups": remaining}

    # ========== 会话生命周期(场景共享) ==========
    def cancel_video_propagate(self, session_id: str) -> Dict:
        """
        请求取消正在进行的 submit 传播(协作式取消)
        只是置位 cancel_event, 当前帧算完后循环自行退出并 yield cancelled
        """
        session = self._get_video_session(session_id)
        session.cancel_event.set()
        return {"success": True, "message": "取消信号已发出, 当前帧算完后停止"}

    def reset_video_tracking(self, session_id: str) -> Dict:
        """
        清空追踪状态(提示/物体/缓存/前沿), 但保留帧入库状态
        (流式的 stream_c2s/s2c 映射、received_frame_count、last_frame 不动,
        已入库的帧不需要重推; 底层用 clear_objects 全量清空追踪记忆)
        之后需要重新加提示并 submit
        重来一次分割, 不是重来一遍视频, session依旧存在, 这个使用场景是在流视频
        """
        session = self._get_video_session(session_id)
        session.cancel_event.set()          # 若有正在进行的 submit, 一并停掉
        self.compute_engine.clear_video_objects(session.video_session)
        session.frame_prompts.clear()
        session.dirty_prompts.clear()
        session.keyframe_results.clear()
        session.submitted_groups.clear()
        session.group_first_frame.clear()
        session.anchor_frame = None
        session.last_computed_frame = None
        return {"success": True, "message": "追踪状态已重置, 帧入库状态保留"}

    def close_video_session(self, session_id: str) -> Dict:
        """
        删除整个视频会话(先发出取消信号停掉可能的在途 submit)
        engine close_session 原语释放外部资源(disk帧仓段文件),
        底层 inference_session 本体随引用释放
        """
        session = self.session_manager.get_video_session(session_id)
        if session is None:
            raise ValueError(f"会话 {session_id} 不存在")
        session.cancel_event.set() # 停掉submit
        self.compute_engine.close_video_session(session.video_session) # 删除帧仓段文件
        self.session_manager.delete_video_session(session_id) # 删除视频会话
        return {"success": True, "message": f"会话 {session_id} 已删除"}

    # ---- 流式推帧原语(仅流式会话的合法路径, 由 StreamingVideoService 覆写) ----
    def _push_streaming_frame(self, session, frame, track: bool = True) -> Dict:
        """离线实例到达此处即场景路由错误(正常路径被 is_streaming 防护拦截)"""
        raise NotImplementedError("_push_streaming_frame 仅流式会话支持")

    # ========== 会话创建(双场景入口, 依 chunk_iter 分流) ==========
    # 这个写法不在于videos_frames是来自于内存还是硬盘, 区分点还是在于流式还是离线
    def create_video_session(self, video_frames=None, frame_stride=1,
                         auto_predict=True, chunk_size=32) -> str:
        """离线(帧已在内存)/流式(传None)"""
        if video_frames is None:
            return self._create_from_chunks(None, 0, frame_stride, auto_predict)
        num = len(video_frames)
        chunks = (video_frames[s:s + chunk_size] for s in range(0, num, chunk_size))
        return self._create_from_chunks(chunks, num, frame_stride, auto_predict) # 这样是为了解决显存的OOM问题

    def _create_from_chunks(self, chunk_iter, num_frames, frame_stride, auto_predict) -> str:
        """
        唯一实现:注册 + 分批入库
        离线: 帧仓落盘(safetensors分段), 长视频不占RAM; 流式: 保留内存dict
        为应对帧的序号在流式的情况下出现乱序的情况, 流式将来要加乱序重排缓冲, 依赖dict的随机写能力
        """
        use_disk = chunk_iter is not None
        compute_session = self.compute_engine.init_video_session(
            None, video_storage_device='cpu',
            frame_store="disk" if use_disk else "ram")
        compute_session["num_frames"] = num_frames
        sid = self.session_manager.register_video_session(
            compute_session, chunk_iter is None,
            frame_stride=frame_stride, auto_predict=auto_predict
        )
        if chunk_iter is not None:
            session = self.session_manager.get_video_session(sid) # 开始将帧分批入库
            for chunk in chunk_iter:
                out = self.compute_engine.add_video_frames(session.video_session, chunk)
                session.original_size = out["original_size"]
        return sid

    # ========== 传播(批量提交, 双场景: 内部按 is_streaming 取帧范围) ==========
    def submit_video_prompts(self, session_id: str,
                             start_frame: Optional[int] = None,
                             end_frame: Optional[int] = None,
                             num_frames: Optional[int] = None) -> Generator[Dict, None, None]:
        """
        提交分割申请(生成器, 逐关键帧产出事件, 路由层逐个转发)

        范围: [start_frame, end_frame], 也可用 num_frames 只算一段
        默认 start = 跟踪前沿+1(增量续算), end = 最后一帧
        提示帧自动成为关键帧; 起算点之前的提示帧先补算(lead), 让提示进入追踪记忆
        """
        session = self._get_video_session(session_id)
        max_frame = (session.received_frame_count - 1) if session.is_streaming \
            else session.frame_count - 1
        if max_frame < 0:
            raise ValueError("还没有可计算的帧")

        # 确定计算起点 f0, 三级回退: 手动指定 > 接着上次算 > 从头算
        if start_frame is not None:
            # 1. 用户显式指定: 拖到某帧起新链 / 强制从某帧重算
            f0 = start_frame
        elif session.last_computed_frame is not None:
            # 2.  增量续算: 从跟踪前沿的下一帧开始, 不重复劳动
            #    (删除提示后前沿会回退, 这里自动从作废处重算)
            f0 = session.last_computed_frame + 1
        else:
            # 3.  全新会话首次提交: 从网格锚点开始, 无锚点则从视频开头
            f0 = session.anchor_frame if session.anchor_frame is not None else 0
        end = max_frame if end_frame is None else min(end_frame, max_frame) # 如果不指定end_frame, 则用max_frame, 分割到视频最后一帧, 由会话状态算出来的(内部)

        if num_frames is not None:
            end = min(end, f0 + num_frames - 1) # num_frames是调用方传进来的, 这次调用只想算多少帧(用户参数)
        if f0 > end:
            raise ValueError(f"计算范围为空: start={f0}, end={end}") # 开始大于结束

        prompt_frames = sorted(f for f, groups in session.frame_prompts.items()
                               if any(g.points or g.box is not None for g in groups.values())) # 将提示的帧按照帧的idx进行偏序,再送入计算

        if session.anchor_frame is None:
            session.anchor_frame = f0
        anchor = session.anchor_frame

        # TODO(v2 候选优化): 插帧网格改为"提示对齐"——每次出现新的提示帧后,
        # 网格相位重锚到最新提示帧, 而不是现在的固定锚(首次计算帧)+提示事件帧并集
        #
        # 示例(stride=5, 提示在 3、20 帧, 计算范围 0~30):
        #   现设计关键帧 = 0,3,5,10,15,20,25,30  (锚 0 的网格 ∪ 提示帧)
        #   提示对齐   = 3,8,13,18,20,25,30      (锚 3 的网格, 提示帧 20 后相位不变)
        # 计算量相近, 但后者每个计算帧离最近提示帧 <= stride, 掩码漂移更小
        # (v1 已具备"提示帧后的帧复用提示帧结果"——提示帧本身在并集里, 此处是进一步收紧)
        #
        # 注意点:
        #   1. 流式下可能反而更密: 旧相位帧已算完撤不回, 新相位继续算,
        #      两相位并集使关键帧更密(除非作废旧缓存重算, 代价更大)
        #   2. anchor 变成动态状态(每次新提示帧更新), 影响所有引用处:
        #      _invalidate_from / _latest_cached_before / 本方法的 plan 相位计算
        #      以及 reset_video_tracking 对 anchor 的清理
        #   3. 删除提示帧后相位是否回退需要定义清楚
        # 结论: v1 保留固定锚+事件帧(行为已正确), 此为 v2 候选, 改动前需过一遍上面三点

        # 计算计划 = 网格帧 ∪ 范围内的提示帧; lead = 起点之前的提示帧(先补算, 让提示进入记忆)
        plan = sorted({f for f in range(f0, end + 1) if (f - anchor) % session.frame_stride == 0}
                      | {f for f in prompt_frames if f0 <= f <= end})
        lead = [f for f in prompt_frames if f < f0] # 当显式指定的start_frame越过了某些提示帧时, 比如提示在第3帧, 但是只要10-20帧的结果， 这样的话第3帧的提示就无法输入了, 所以要加lead

        # 取消传播计算
        session.cancel_event.clear()
        yield {"type": "propagate_start", "start_frame": f0, "end_frame": end,
               "num_keyframes": len(plan)}

        # lead一定在plan前面, 因为lead < f0 <= plan
        for f in lead:
            # lead不能太稠密, 不过也不会很稠密, 极端情况暂停模式下用户在几十上百个帧各加了提示，再一次 submit, 不过这种情况下用户已经不在意
            # 实时的这个需求了, 处理一段时间也就处理一段时间了
            if session.cancel_event.is_set():
                yield {"type": "cancelled", "frame_idx": f}
                return
            self._ensure_prompt_frame_pushed(session, f)
            self._flush_dirty_frame(session, f)
            self._compute_and_cache(session, f)
        if lead:
            yield {"type": "prompts_applied", "frames": lead}

        for i, f in enumerate(plan):
            if session.cancel_event.is_set():
                yield {"type": "cancelled", "frame_idx": f, "progress": i / len(plan)}
                return
            self._ensure_prompt_frame_pushed(session, f)
            self._flush_dirty_frame(session, f)
            res = self._compute_and_cache(session, f)
            yield {"type": "keyframe", "frame_idx": f, "progress": (i + 1) / len(plan),
                   **self._visible_result(session, f, res)}

        yield {"type": "propagate_done", "start_frame": f0, "end_frame": end}

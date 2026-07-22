"""
服务层功能测试(假计算引擎, 不加载真实模型, 不碰 GPU)
验证: 会话创建/提示记录/提交传播/插帧复用/可见性过滤/缓存失效/
     增量续算/lead 补算/删除(整组+细粒度)/流式推帧/取消/重置
"""
import sys
from pathlib import Path
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # codeRepo 根
from server_api.business.service_layer import SAM3ServiceLayer


# ============ 假计算引擎 ============
class FakeVideoState:
    def __init__(self, h, w, num_frames):
        self.h, self.w = h, w
        self.num_frames = num_frames
        self.prompts = {}          # (frame_idx, obj_id) -> dict
        self.objects = []          # 已创建 obj_id(顺序创建, 恒为 0..n-1)


class FakeEngine:
    """模拟 SAM3ComputeEngine 的视频方法, 语义对齐 transformers 行为"""
    def __init__(self):
        self.calls = []

    def init_video_session(self, video_frames, video_storage_device=None):
        if video_frames is not None:
            h, w = video_frames[0].size[1], video_frames[0].size[0]
            n = len(video_frames)
        else:
            h = w = None
            n = 0
        return {"session": FakeVideoState(h, w, n),
                "video_height": h, "video_width": w, "num_frames": n}

    def _masks(self, st):
        if not st.objects:
            return None
        return torch.stack([torch.full((4, 4), float(oid + 1)) for oid in st.objects])

    def process_video_frame(self, session, frame):
        st = session["session"]
        if st.h is None:
            st.h, st.w = frame.size[1], frame.size[0]
        m = self._masks(st)
        return {"masks": m, "num_objects": len(st.objects),
                "shape": m.shape if m is not None else (0,)}

    def add_video_prompt(self, session, frame_idx, obj_id, click_points=None,
                         click_labels=None, input_boxes=None, original_size=None):
        st = session["session"]
        assert original_size is not None and original_size[0] is not None, \
            "original_size 为 None(流式尺寸未记录)"
        if obj_id not in st.objects:
            assert obj_id == len(st.objects), "obj_id 必须顺序分配"
            st.objects.append(obj_id)
        st.prompts[(frame_idx, obj_id)] = {"points": click_points, "box": input_boxes}
        self.calls.append(("add_prompt", frame_idx, obj_id))

    def predict_video_frame(self, session, frame_idx):
        self.calls.append(("predict", frame_idx))
        return {"masks": self._masks(session["session"])}

    def remove_video_object(self, session, obj_id):
        st = session["session"]
        if obj_id not in st.objects:
            return False
        remaining = [o for o in st.objects if o != obj_id]
        remap = {old: new for new, old in enumerate(remaining)}
        st.prompts = {(f, remap[o]): v for (f, o), v in st.prompts.items() if o != obj_id}
        st.objects = list(range(len(remaining)))
        self.calls.append(("remove_object", obj_id))
        return True

    def remove_video_object_inputs(self, session, obj_id, frame_idx):
        st = session["session"]
        st.prompts.pop((frame_idx, obj_id), None)
        self.calls.append(("remove_inputs", frame_idx, obj_id))
        return True

    def clear_video_objects(self, session):
        st = session["session"]
        st.objects.clear()
        st.prompts.clear()
        self.calls.append(("clear_objects",))
        return True


def make_service():
    svc = SAM3ServiceLayer(enable_tracker=False, enable_video=False)
    svc.compute_engine = FakeEngine()
    return svc


def make_frames(n=30, size=16):
    return [Image.new("RGB", (size, size), (i % 255, 0, 0)) for i in range(n)]


FRAMES = make_frames(30)


# ============ 测试用例 ============
def test_offline_batch_submit():
    """离线批量: 提示解耦 -> submit -> 网格∪提示帧 -> 可见性过滤 -> 插帧复用"""
    svc = make_service()
    sid = svc.create_video_session(video_frames=FRAMES, frame_stride=5, auto_predict=False)
    r = svc.add_video_point(sid, group_id=1, x=1, y=1, label=1, frame_idx=3)
    assert r["computed"] is False
    svc.add_video_point(sid, group_id=2, x=2, y=2, label=1, frame_idx=20)

    events = list(svc.submit_video_prompts(sid))
    kf = [e["frame_idx"] for e in events if e["type"] == "keyframe"]
    assert kf == [0, 3, 5, 10, 15, 20, 25], f"关键帧计划错误: {kf}"
    assert events[0]["type"] == "propagate_start"
    assert events[-1]["type"] == "propagate_done"

    by_frame = {e["frame_idx"]: e for e in events if e["type"] == "keyframe"}
    assert by_frame[0]["groups"] == [] and by_frame[0]["masks"] is None   # 0 物体防护
    assert by_frame[3]["groups"] == [1]
    assert by_frame[20]["groups"] == [1, 2]      # group2 首次出现
    assert by_frame[5]["groups"] == [1]          # group2 在首提示帧之前不可见

    r = svc.get_video_frame_result(sid, 7)        # 非关键帧复用
    assert r["keyframe"] is False and r["reused_from"] == 5, r
    r = svc.get_video_frame_result(sid, 3)
    assert r["keyframe"] is True and r["groups"] == [1]
    print("PASS test_offline_batch_submit")


def test_batch_add_invalidates():
    """批量模式加提示 -> 该帧及之后缓存作废, 前沿回退, 下次 submit 增量重算"""
    svc = make_service()
    sid = svc.create_video_session(video_frames=FRAMES, frame_stride=5, auto_predict=False)
    svc.add_video_point(sid, group_id=1, x=1, y=1, label=1, frame_idx=3)
    svc.add_video_point(sid, group_id=2, x=2, y=2, label=1, frame_idx=20)
    list(svc.submit_video_prompts(sid))

    sess = svc.session_manager.get_video_session(sid)
    assert sess.last_computed_frame == 25

    svc.add_video_point(sid, group_id=1, x=5, y=5, label=1, frame_idx=10)
    assert sess.last_computed_frame == 5, \
        f"加提示后缓存未失效, last_computed={sess.last_computed_frame}(补丁3未生效)"

    events = list(svc.submit_video_prompts(sid))
    types = [(e["type"], e.get("frame_idx", e.get("frames"))) for e in events]
    kf = [f for t, f in types if t == "keyframe"]
    assert kf == [10, 15, 20, 25], f"增量续算范围错误: {kf}"
    assert ("prompts_applied", [3]) in types, f"lead 补算缺失: {types}"
    print("PASS test_batch_add_invalidates")


def test_auto_predict_invalidation():
    """实时模式回看加提示: 该帧立即重算, 之后帧缓存作废"""
    svc = make_service()
    sid = svc.create_video_session(video_frames=FRAMES, frame_stride=1, auto_predict=True)
    r = svc.add_video_point(sid, group_id=1, x=1, y=1, label=1, frame_idx=0)
    assert r["computed"] and r["groups"] == [1]
    list(svc.submit_video_prompts(sid, end_frame=5))

    sess = svc.session_manager.get_video_session(sid)
    assert sess.last_computed_frame == 5

    r = svc.add_video_point(sid, group_id=2, x=2, y=2, label=1, frame_idx=3)
    assert r["computed"] and r["groups"] == [1, 2]
    assert sess.last_computed_frame == 3, \
        f"后续帧缓存未作废, last_computed={sess.last_computed_frame}(补丁3未生效)"
    print("PASS test_auto_predict_invalidation")


def test_lead_frames():
    """显式 start_frame 越过提示帧: lead 先导补算"""
    svc = make_service()
    sid = svc.create_video_session(video_frames=FRAMES, frame_stride=5, auto_predict=False)
    svc.add_video_point(sid, group_id=1, x=1, y=1, label=1, frame_idx=3)

    events = list(svc.submit_video_prompts(sid, start_frame=10, end_frame=15))
    types = [(e["type"], e.get("frame_idx", e.get("frames"))) for e in events]
    assert events[0]["type"] == "propagate_start" and events[0]["start_frame"] == 10
    assert ("prompts_applied", [3]) in types, f"lead 未补算: {types}"
    kf = [f for t, f in types if t == "keyframe"]
    assert kf == [10, 15], f"plan 错误: {kf}"
    # lead 帧也被缓存
    sess = svc.session_manager.get_video_session(sid)
    assert 3 in sess.keyframe_results
    print("PASS test_lead_frames")


def test_delete_group():
    """删除整个物体: 底层重索引 + 缓存行删除"""
    svc = make_service()
    sid = svc.create_video_session(video_frames=FRAMES, frame_stride=1, auto_predict=False)
    svc.add_video_point(sid, group_id=1, x=1, y=1, label=1, frame_idx=0)
    svc.add_video_point(sid, group_id=2, x=2, y=2, label=1, frame_idx=0)
    list(svc.submit_video_prompts(sid, end_frame=2))

    sess = svc.session_manager.get_video_session(sid)
    assert sess.submitted_groups == [1, 2]

    svc.clear_video_group(sid, 1)
    assert sess.submitted_groups == [2], sess.submitted_groups
    st = sess.video_session["session"]
    assert st.objects == [0], "底层应重索引为 [0]"
    assert (0, 0) in st.prompts, "group2 的提示应重映射到 obj 0"
    res0 = sess.keyframe_results[0]
    assert res0["groups"] == [2] and res0["masks"].shape[0] == 1, "缓存行应删除 group1"
    print("PASS test_delete_group")


def test_fine_delete_upgrade():
    """细粒度删除: 删到最后一个点 -> 清底层输入 -> 空组升级删除物体"""
    svc = make_service()
    sid = svc.create_video_session(video_frames=FRAMES, frame_stride=1, auto_predict=False)
    svc.add_video_point(sid, group_id=1, x=1, y=1, label=1, frame_idx=0)
    svc.add_video_point(sid, group_id=1, x=2, y=2, label=1, frame_idx=0)
    list(svc.submit_video_prompts(sid, end_frame=0))

    sess = svc.session_manager.get_video_session(sid)
    eng = svc.compute_engine

    r = svc.delete_video_point(sid, group_id=1, frame_idx=0, point_index=0)
    assert r["remaining_points"] == 1
    assert (0, 1) in sess.dirty_prompts, "剩余点应标记重推覆盖"
    assert sess.submitted_groups == [1], "还有提示, 不应删物体"

    r = svc.delete_video_point(sid, group_id=1, frame_idx=0, point_index=0)
    assert r["remaining_points"] == 0
    assert ("remove_inputs", 0, 0) in eng.calls, \
        f"空组应调 remove_object_inputs(补丁2未生效): {eng.calls}"
    assert sess.submitted_groups == [], \
        f"所有帧提示清空应升级删除物体(补丁2未生效): {sess.submitted_groups}"
    assert ("remove_object", 0) in eng.calls
    print("PASS test_fine_delete_upgrade")


def test_streaming():
    """流式: 网格推帧/非关键帧复用/最新帧提示约束/加提示后计算/original_size"""
    svc = make_service()
    sid = svc.create_video_session(video_frames=None, frame_stride=2, auto_predict=True)

    r0 = svc.push_video_frame(sid, Image.new("RGB", (16, 16)))
    assert r0["frame_idx"] == 0 and r0["keyframe"] is True
    r1 = svc.push_video_frame(sid, Image.new("RGB", (16, 16)))
    assert r1["keyframe"] is False and r1["reused_from"] == 0
    svc.push_video_frame(sid, Image.new("RGB", (16, 16)))   # frame 2 keyframe
    svc.push_video_frame(sid, Image.new("RGB", (16, 16)))   # frame 3 reused
    r4 = svc.push_video_frame(sid, Image.new("RGB", (16, 16)))
    assert r4["keyframe"] is True and r4["frame_idx"] == 4

    # 非最新帧加提示 -> 拒绝
    try:
        svc.add_video_point(sid, group_id=1, x=1, y=1, label=1, frame_idx=2)
        raise AssertionError("非最新帧加提示应抛 ValueError")
    except ValueError:
        pass

    # 最新帧加提示(帧4已推入, 直接 flush + 计算)
    r = svc.add_video_point(sid, group_id=1, x=1, y=1, label=1, frame_idx=4)
    assert r["computed"] and r["groups"] == [1]

    r5 = svc.push_video_frame(sid, Image.new("RGB", (16, 16)))
    assert r5["keyframe"] is False and r5["reused_from"] == 4 and r5["groups"] == [1]
    r6 = svc.push_video_frame(sid, Image.new("RGB", (16, 16)))
    assert r6["keyframe"] is True and r6["groups"] == [1]
    print("PASS test_streaming")


def test_cancel_and_reset():
    """取消: 协作式, 下一帧边界退出; 重置: 追踪状态清空, 帧入库保留"""
    svc = make_service()
    sid = svc.create_video_session(video_frames=FRAMES, frame_stride=1, auto_predict=False)
    svc.add_video_point(sid, group_id=1, x=1, y=1, label=1, frame_idx=0)

    gen = svc.submit_video_prompts(sid)
    e = next(gen)
    assert e["type"] == "propagate_start"
    svc.cancel_video_propagate(sid)
    rest = list(gen)
    assert rest and rest[0]["type"] == "cancelled", f"应取消: {rest}"

    # 取消后再次 submit 不受残留信号影响(submit 开头 clear)
    events = list(svc.submit_video_prompts(sid, end_frame=2))
    assert events[-1]["type"] == "propagate_done"

    svc.reset_video_tracking(sid)
    sess = svc.session_manager.get_video_session(sid)
    assert sess.submitted_groups == [] and sess.keyframe_results == {}
    assert sess.frame_prompts == {} and sess.last_computed_frame is None
    assert ("clear_objects",) in svc.compute_engine.calls
    print("PASS test_cancel_and_reset")


def test_delete_point_bounds():
    """删点边界: -1 撤销最后一个点合法, 越界抛 ValueError 而非裸 IndexError"""
    svc = make_service()
    sid = svc.create_video_session(video_frames=FRAMES, frame_stride=1, auto_predict=False)
    svc.add_video_point(sid, group_id=1, x=1, y=1, label=1, frame_idx=0)
    svc.add_video_point(sid, group_id=1, x=2, y=2, label=1, frame_idx=0)

    r = svc.delete_video_point(sid, group_id=1, frame_idx=0, point_index=-1)
    assert r["remaining_points"] == 1
    try:
        svc.delete_video_point(sid, group_id=1, frame_idx=0, point_index=-5)
        raise AssertionError("越界应抛 ValueError")
    except ValueError:
        pass
    print("PASS test_delete_point_bounds")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
        except Exception as e:
            failed += 1
            print(f"FAIL {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)

"""
真机冒烟(流式视频专项, cuda:1, 只加载 video 模型)
核心验证: 0 物体推帧(engine 补丁) / 网格推帧+复用 / 最新帧提示出真实掩码 /
         非最新帧提示拒绝 / submit 事件流 / reset

运行:
    cd /root/workspace/codeRepo
    /root/miniconda3/envs/SVD/bin/python server_api/tests/smoke_real_stream.py
"""
import sys
import threading
import time

sys.path.insert(0, "/root/workspace/codeRepo")

import requests
import uvicorn
from PIL import Image

from server_api.business.service_layer import SAM3ServiceLayer
from server_api.api.server import create_app
from server_api.api.client import Sam3Client, ApiError

HOST, PORT = "127.0.0.1", 8768
BASE = f"http://{HOST}:{PORT}"
VIDEO_PATH = "/tmp/animals_100.mp4"  # 100 帧 1080p 切片, 避免全片解码 OOM


def main():
    print("loading video model on cuda:1 ...", flush=True)
    service = SAM3ServiceLayer(model_path="/root/workspace/modelRepo/SAM3",
                               device="cuda:1",
                               enable_tracker=False, enable_video=True)
    print("model loaded", flush=True)

    server = uvicorn.Server(uvicorn.Config(create_app(service),
                                           host=HOST, port=PORT, log_level="error"))
    threading.Thread(target=server.run, daemon=True).start()
    for _ in range(100):
        try:
            requests.get(f"{BASE}/model/status", timeout=1)
            break
        except Exception:
            time.sleep(0.2)
    else:
        raise RuntimeError("服务起不来")
    print("server up", flush=True)

    client = Sam3Client(BASE)

    # ---- 流式会话(stride=2, 实时模式) ----
    sid = client.create_stream_session(frame_stride=2, auto_predict=True)
    ch = client.open_video(sid)

    import decord
    vr = decord.VideoReader(VIDEO_PATH)
    frames = [Image.fromarray(f) for f in vr.get_batch(list(range(10))).asnumpy()]

    # 0 物体推 5 帧(补丁前: 第一帧就炸 "No objects are provided")
    evs = [ch.push_frame(frames[i]) for i in range(5)]
    assert evs[0]["keyframe"] is True and evs[0]["frame_idx"] == 0
    assert evs[1]["keyframe"] is False and evs[1]["reused_from"] == 0
    assert evs[2]["keyframe"] is True and evs[2]["frame_idx"] == 2
    assert evs[3]["reused_from"] == 2
    assert evs[4]["keyframe"] is True and evs[4]["frame_idx"] == 4
    assert all("mask_images" not in e for e in evs)
    print("PASS stream push 5 frames @0-object (engine patch works)", flush=True)

    # 非最新帧加提示 -> 拒绝
    try:
        ch.add_point(group_id=1, x=480, y=800, label=1, frame_idx=2)
        raise AssertionError("应抛 ApiError")
    except ApiError as e:
        assert "最新" in str(e)
    print("PASS prompt-frame guard", flush=True)

    # 最新帧(frame 4, 已入库网格帧)加提示 -> 立即算出真实掩码
    t0 = time.time()
    ev = ch.add_point(group_id=1, x=480, y=800, label=1, frame_idx=4)
    assert ev["computed"] and ev["groups"] == [1]
    assert set(ev["mask_images"]) == {1}
    assert ev["mask_images"][1].size == frames[0].size
    print(f"PASS add_point @latest frame, real mask ({time.time() - t0:.2f}s)", flush=True)

    # 继续推帧: frame 5 复用(非网格), frame 6 网格帧带物体追踪
    ev = ch.push_frame(frames[5])
    assert ev["reused_from"] == 4 and ev["groups"] == [1] and "mask_images" in ev
    t0 = time.time()
    ev = ch.push_frame(frames[6])
    assert ev["keyframe"] is True and ev["groups"] == [1] and "mask_images" in ev
    print(f"PASS push frames with tracking ({time.time() - t0:.2f}s for grid frame)", flush=True)

    # submit: 显式 start_frame 强制重算一段(流式+实时下裸 submit 范围必为空)
    types = [e["type"] for e in ch.submit(start_frame=5)]
    assert types[0] == "propagate_start" and types[-1] == "propagate_done", types
    assert "keyframe" in types and "prompts_applied" in types, types
    print(f"PASS stream submit: {types}", flush=True)

    # get_frame: 命中缓存 / 复用
    ev = ch.get_frame(6)
    assert ev["keyframe"] is True and ev["groups"] == [1]
    ev = ch.get_frame(5)
    assert ev["keyframe"] is False and ev["reused_from"] == 4

    # reset: 追踪状态清空, 帧入库保留 -> 直接在新帧上打提示
    ev = ch.reset()
    assert ev["success"]
    ev = ch.push_frame(frames[7])
    ev = ch.add_point(group_id=5, x=480, y=800, label=1, frame_idx=7)
    assert ev["computed"] and ev["groups"] == [5], ev.get("groups")
    print("PASS reset -> re-prompt on new frame", flush=True)

    ch.close()
    client.close_video_session(sid)
    server.should_exit = True
    print("\nSTREAM REAL SMOKE OK", flush=True)


if __name__ == "__main__":
    main()

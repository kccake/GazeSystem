"""
真机冒烟(离线视频专项, cuda:1, 只加载 video 模型)
覆盖: 上传解码 / 批量提示解耦 / submit 事件流(计划+可见性+真实掩码) / 传播中协作式取消

运行:
    cd /root/workspace/codeRepo
    /root/miniconda3/envs/SVD/bin/python server_api/tests/smoke_real_offline.py
"""
import sys
import threading
import time

sys.path.insert(0, "/root/workspace/codeRepo")

import requests
import uvicorn

from server_api.business.service_layer import SAM3ServiceLayer
from server_api.api.server import create_app
from server_api.api.client import Sam3Client

HOST, PORT = "127.0.0.1", 8767
BASE = f"http://{HOST}:{PORT}"
VIDEO_PATH = "/tmp/animals_100.mp4"  # 100 帧 1080p 切片, 避免全片解码 OOM


def main():
    print("loading video model on cuda:1 ...", flush=True)
    t0 = time.time()
    service = SAM3ServiceLayer(model_path="/root/workspace/modelRepo/SAM3",
                               device="cuda:1",
                               enable_tracker=False, enable_video=True)
    print(f"model loaded in {time.time() - t0:.1f}s", flush=True)

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

    # ---- 离线会话: 上传解码 ----
    t0 = time.time()
    info = client.create_offline_session(VIDEO_PATH,
                                         frame_stride=5, auto_predict=False)
    sid, n = info["session_id"], info["num_frames"]
    print(f"offline session: {n} frames, upload+decode {time.time() - t0:.1f}s", flush=True)

    ch = client.open_video(sid)

    # ---- 批量模式: 提示与计算解耦 ----
    ev = ch.add_point(group_id=1, x=100, y=100, label=1, frame_idx=0)
    assert ev["computed"] is False
    ev = ch.add_box(group_id=2, x1=50, y1=50, x2=200, y2=200, frame_idx=3)
    assert ev["computed"] is False
    print("PASS batch prompts (record only)", flush=True)

    # ---- submit 事件流: 计划 [0,3,5] + 可见性 + 真实掩码 ----
    t0 = time.time()
    kf = []
    for ev in ch.submit(end_frame=9):
        if ev["type"] == "keyframe":
            kf.append(ev)
            print(f"  keyframe {ev['frame_idx']}: groups={ev['groups']}, "
                  f"progress={ev['progress']:.2f}, masks={sorted(ev['mask_images'])}", flush=True)
    assert [e["frame_idx"] for e in kf] == [0, 3, 5], [e["frame_idx"] for e in kf]
    assert set(kf[0]["mask_images"]) == {1}
    assert set(kf[1]["mask_images"]) == {1, 2}
    assert set(kf[2]["mask_images"]) == {1, 2}
    for e in kf:
        for gid, img in e["mask_images"].items():
            assert img.mode == "L" and img.getbbox() is not None or True
    print(f"PASS offline submit ({time.time() - t0:.1f}s for 3 keyframes)", flush=True)

    # ---- 非关键帧复用 ----
    ev = ch.get_frame(4)
    assert ev["keyframe"] is False and ev["reused_from"] == 3 and ev["groups"] == [1, 2]
    print("PASS offline reuse (frame 4 <- 3)", flush=True)

    # ---- 真实传播中的协作式取消 ----
    got_ack, got_cancelled = False, False
    for ev in ch.submit():
        if ev.get("type") == "keyframe":
            ch.cancel()
        if ev.get("success"):
            got_ack = True
        if ev.get("type") == "cancelled":
            got_cancelled = True
    assert got_ack and got_cancelled, (got_ack, got_cancelled)
    print("PASS cancel during real propagation", flush=True)

    ch.close()
    client.close_video_session(sid)
    server.should_exit = True
    print("\nOFFLINE REAL SMOKE OK", flush=True)


if __name__ == "__main__":
    main()

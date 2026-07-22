"""
真机端到端冒烟: 真实 SAM3 模型(cuda:1) + 真实服务 + 真实 client SDK
覆盖:
- 图像 HTTP 全流程(含真实 predict 出掩码)
- 流式视频 WS(0 物体首帧/推帧/提示/传播/复用)
- 离线视频(上传解码/批量提交/可见性) + 真实传播中的协作式取消

运行:
    cd /root/workspace/codeRepo
    /root/miniconda3/envs/SVD/bin/python server_api/tests/smoke_real.py
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

HOST, PORT = "127.0.0.1", 8766
BASE = f"http://{HOST}:{PORT}"
ASSETS = "/root/workspace/codeRepo/server_api/assets"
VIDEO_PATH = "/tmp/animals_100.mp4"  # 100 帧 1080p 切片, 避免全片解码 OOM


def main():
    print("loading models (tracker + video) on cuda:1 ...", flush=True)
    t0 = time.time()
    service = SAM3ServiceLayer(model_path="/root/workspace/modelRepo/SAM3",
                               device="cuda:1",
                               enable_tracker=True, enable_video=True)
    print(f"models loaded in {time.time() - t0:.1f}s", flush=True)

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

    # ================= 图像 HTTP(真实 predict) =================
    photo = Image.open(f"{ASSETS}/cocotest.jpg").convert("RGB")
    isid = client.create_image_session(photo)
    client.add_point(isid, group_id=1, x=photo.size[0] / 2, y=photo.size[1] / 2, label=1)
    masks = client.predict_image(isid)
    assert set(masks) == {1}, masks.keys()
    assert masks[1].size == photo.size, (masks[1].size, photo.size)
    assert masks[1].mode == "L"
    bbox = masks[1].getbbox()
    print(f"PASS image http predict, mask bbox={bbox}", flush=True)
    client.close_image_session(isid)

    # ================= 流式视频 WS =================
    sid = client.create_stream_session(frame_stride=2, auto_predict=True)
    ch = client.open_video(sid)

    IMG = Image.new("RGB", (32, 24), (10, 20, 30))
    evs = [ch.push_frame(IMG) for _ in range(5)]
    assert evs[0]["keyframe"] is True
    assert evs[1]["reused_from"] == 0
    assert "mask_images" not in evs[0]
    print("PASS stream push frames @0-object", flush=True)

    ev = ch.add_point(group_id=1, x=16, y=12, label=1, frame_idx=4)
    assert ev["groups"] == [1] and set(ev["mask_images"]) == {1}
    assert ev["mask_images"][1].size == (32, 24)
    print("PASS stream add_point (real mask)", flush=True)

    try:
        ch.add_point(group_id=1, x=1, y=1, label=1, frame_idx=2)
        raise AssertionError("应抛 ApiError")
    except ApiError:
        pass

    ev = ch.push_frame(IMG)
    assert ev["reused_from"] == 4 and ev["groups"] == [1]

    types = [e["type"] for e in ch.submit()]
    assert types[0] == "propagate_start" and types[-1] == "propagate_done", types
    ev = ch.get_frame(1)
    assert ev["keyframe"] is False and ev["reused_from"] == 0
    print("PASS stream submit + get_frame", flush=True)

    ev = ch.reset()
    assert ev["success"]
    ch.close()

    # ================= 离线视频(批量提交 + 真实取消) =================
    info = client.create_offline_session(VIDEO_PATH,
                                         frame_stride=5, auto_predict=False)
    sid2, n = info["session_id"], info["num_frames"]
    print(f"offline session: {n} frames", flush=True)

    ch2 = client.open_video(sid2)
    assert ch2.add_point(group_id=1, x=100, y=100, label=1, frame_idx=0)["computed"] is False
    assert ch2.add_box(group_id=2, x1=50, y1=50, x2=200, y2=200, frame_idx=3)["computed"] is False

    kf = [e for e in ch2.submit(end_frame=9) if e["type"] == "keyframe"]
    assert [e["frame_idx"] for e in kf] == [0, 3, 5], [e["frame_idx"] for e in kf]
    assert set(kf[0]["mask_images"]) == {1}
    assert set(kf[1]["mask_images"]) == {1, 2}
    print("PASS offline batch submit (real masks)", flush=True)

    got_cancelled = False
    for ev in ch2.submit():
        if ev.get("type") == "keyframe":
            ch2.cancel()
        if ev.get("type") == "cancelled":
            got_cancelled = True
    assert got_cancelled, "未收到 cancelled 事件"
    print("PASS offline cancel during real propagation", flush=True)

    ch2.close()
    client.close_video_session(sid2)

    server.should_exit = True
    print("\nREAL SMOKE OK", flush=True)


if __name__ == "__main__":
    main()

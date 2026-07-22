"""
真机冒烟(prompt 文件导入专项, cuda:1, 同时加载 image+video 模型)
覆盖:
  图像(HTTP): load_image_prompt_file 的 append/skip/replace + predict 出掩码
  视频(WS):    load_prompt_file 批量导入多帧提示 -> submit 传播出掩码
               组可见性(第 10 帧才出现的组, 之前帧不可见) / skip 跳过

运行:
    cd /root/workspace/codeRepo
    /root/miniconda3/envs/SVD/bin/python GazeSystem/tests/smoke_prompt_file.py
"""
import sys
import threading
import time

sys.path.insert(0, "/root/workspace/codeRepo")

import requests
import uvicorn

from GazeSystem.business.service_layer import SAM3ServiceLayer
from GazeSystem.api.server import create_app
from GazeSystem.api.client import Sam3Client

HOST, PORT = "127.0.0.1", 8768
BASE = f"http://{HOST}:{PORT}"
VIDEO_PATH = "/tmp/animals_30.mp4"  # 30 帧 1080p 切片(GPU 紧张, 降低预处理显存峰值)

PASS = []


def check(name: str, cond: bool, detail: str = ""):
    PASS.append(cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""),
          flush=True)


def grab_frame0():
    """取视频第 0 帧当测试图像(decord 必须在 torch 之后导入)"""
    import decord
    vr = decord.VideoReader(VIDEO_PATH)
    from PIL import Image
    return Image.fromarray(vr[0].asnumpy())


def main():
    print("loading image+video models on cuda:0 ...", flush=True)
    t0 = time.time()
    service = SAM3ServiceLayer(model_path="/root/workspace/modelRepo/SAM3",
                               device="cuda:0",
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

    # ================= 图像: HTTP prompt_file =================
    print("\n== image prompt_file ==", flush=True)
    img = grab_frame0()
    sid = client.create_image_session(img)

    img_file = {
        "type": "image",
        "groups": [
            {"group_id": 1, "points": [[100, 100]], "labels": [1]},
            {"group_id": 2, "points": [[300, 300]], "labels": [1],
             "box": [50, 50, 400, 400]},
        ],
    }

    r = client.load_image_prompt_file(sid, img_file, merge_mode="append")
    check("append 加载 2 组",
          sorted(g["group_id"] for g in r["groups_loaded"]) == [1, 2], str(r))

    masks = client.predict_image(sid)
    check("predict 返回组 1/2 掩码", sorted(masks.keys()) == [1, 2],
          f"keys={sorted(masks.keys())}")

    r = client.load_image_prompt_file(sid, img_file, merge_mode="skip")
    check("skip 模式两组都跳过", sorted(r["groups_skipped"]) == [1, 2], str(r))

    img_file2 = {"type": "image", "groups": [
        {"group_id": 1, "points": [[200, 200]], "labels": [1]}]}
    r = client.load_image_prompt_file(sid, img_file2, merge_mode="replace")
    check("replace 覆盖组 1",
          [g["group_id"] for g in r["groups_loaded"]] == [1], str(r))
    masks = client.predict_image(sid)
    check("replace 后 predict 正常", sorted(masks.keys()) == [1, 2],
          f"keys={sorted(masks.keys())}")

    r = client.load_image_prompt_file(sid, img_file2, merge_mode="append")
    check("append 叠加组 1",
          [g["group_id"] for g in r["groups_loaded"]] == [1], str(r))
    masks = client.predict_image(sid)
    check("append 后 predict 正常", sorted(masks.keys()) == [1, 2],
          f"keys={sorted(masks.keys())}")

    client.close_image_session(sid)

    # ================= 视频: WS prompt_file =================
    print("\n== video prompt_file (WS) ==", flush=True)
    info = client.create_offline_session(VIDEO_PATH, frame_stride=5,
                                         auto_predict=False)
    vsid, n = info["session_id"], info["num_frames"]
    print(f"offline session: {n} frames", flush=True)
    ch = client.open_video(vsid)

    video_file = {
        "type": "video",
        "frames": [
            {"frame_idx": 0, "groups": [
                {"group_id": 1, "points": [[100, 100]], "labels": [1]}]},
            {"frame_idx": 10, "groups": [
                {"group_id": 2, "points": [[300, 300]], "labels": [1],
                 "box": [50, 50, 400, 400]}]},
        ],
    }

    r = ch.load_prompt_file(video_file, merge_mode="append")
    check("append 加载帧 0/10", r["loaded_frames"] == [0, 10], str(r))
    check("无跳过", r["skipped_groups"] == [], str(r))

    # submit: stride=5, 0~14 -> 计划帧 {0,5,10}
    events = list(ch.submit(start_frame=0, num_frames=15))
    kf = {e["frame_idx"]: e for e in events if e.get("type") == "keyframe"}
    check("submit 计划帧 = {0,5,10}", sorted(kf.keys()) == [0, 5, 10],
          f"got {sorted(kf.keys())}")
    g0 = sorted(kf[0].get("mask_images", {}).keys()) if 0 in kf else []
    g10 = sorted(kf[10].get("mask_images", {}).keys()) if 10 in kf else []
    check("帧 0 只见组 1(组 2 第 10 帧才出现)", g0 == [1], f"got {g0}")
    check("帧 10 见组 1/2", g10 == [1, 2], f"got {g10}")

    # skip: 同文件再来一遍, 两组都已存在
    r = ch.load_prompt_file(video_file, merge_mode="skip")
    check("skip 跳过全部 2 组", len(r["skipped_groups"]) == 2, str(r))
    check("skip 后 loaded_frames 为空", r["loaded_frames"] == [], str(r))

    # replace: 换掉帧 0 组 1 的提示, 缓存应从帧 0 起作废 -> get_frame 需重算
    video_file3 = {"type": "video", "frames": [
        {"frame_idx": 0, "groups": [
            {"group_id": 1, "points": [[200, 200]], "labels": [1]}]}]}
    r = ch.load_prompt_file(video_file3, merge_mode="replace")
    check("replace 加载帧 0", r["loaded_frames"] == [0], str(r))
    ev = ch.get_frame(0, compute_if_missing=True)
    check("replace 后帧 0 可重算出掩码",
          sorted(ev.get("mask_images", {}).keys()) == [1],
          f"keys={sorted(ev.get('mask_images', {}).keys())}")

    ch.close()
    client.close_video_session(vsid)

    print(f"\n{'ALL PASS' if all(PASS) else 'HAS FAIL'}: "
          f"{sum(PASS)}/{len(PASS)}", flush=True)
    sys.exit(0 if all(PASS) else 1)


if __name__ == "__main__":
    main()

"""
真机冒烟(流式拥塞控制专项, cuda:0, 只加载 video 模型)
验证: 显式帧号协议 / 有界合并缓冲 / 丢弃账恒等式 / flow 观测 /
     空洞帧提示拒绝 / 已入库帧提示出真实掩码 / 旧协议(无帧号头)兼容

核心场景: 25fps fire-and-forget 推帧(发送线程) > 处理吞吐(~12fps),
          服务端合并缓冲必然丢弃中间帧 —— 校验丢弃账恒等式:
          对每个处理结果 j_k: j_k - j_{k-1} - 1 == transport_dropped_k

运行:
    cd /root/workspace/codeRepo
    /root/miniconda3/envs/SVD/bin/python GazeSystem/tests/smoke_stream_flow.py
"""
import sys
import threading
import time

sys.path.insert(0, "/root/workspace/codeRepo")

import requests
import uvicorn
from PIL import Image

from GazeSystem.business.service_layer import SAM3ServiceLayer
from GazeSystem.api.server import create_app
from GazeSystem.api.client import Sam3Client, ApiError
from GazeSystem.api.protocol import encode_image, pack_push_frame

HOST, PORT = "127.0.0.1", 8769
BASE = f"http://{HOST}:{PORT}"
VIDEO_PATH = "/tmp/animals_100.mp4"
N_PUSH, PUSH_INTERVAL = 100, 0.04   # 25fps × 4s, 覆盖 2s 的 stride 评估窗口


def main():
    print("loading video model on cuda:0 ...", flush=True)
    service = SAM3ServiceLayer(model_path="/root/workspace/modelRepo/SAM3",
                               device="cuda:0",
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
    import decord
    vr = decord.VideoReader(VIDEO_PATH)
    frames = [Image.fromarray(f) for f in vr.get_batch(list(range(len(vr)))).asnumpy()]
    assert len(frames) >= N_PUSH, f"视频不足 {N_PUSH} 帧: {len(frames)}"

    # ================= 场景A: 25fps fire-and-forget 过载灌帧 =================
    sid = client.create_stream_session(frame_stride=1, auto_predict=True)
    ch = client.open_video(sid)
    results, recv_done = [], threading.Event()

    def receiver():
        while not recv_done.is_set():
            try:
                ev = ch.recv_event(timeout=30)
            except Exception as e:
                recv_done.set()
                raise RuntimeError(f"接收失败: {e}")
            results.append(ev)
            if ev.get("frame_idx") == N_PUSH - 1:
                recv_done.set()

    def sender():
        for i in range(N_PUSH):
            ch.ws.send(pack_push_frame(i, encode_image(frames[i], "JPEG")))
            time.sleep(PUSH_INTERVAL)

    t0 = time.time()
    rt = threading.Thread(target=receiver, daemon=True)
    rt.start()
    threading.Thread(target=sender, daemon=True).start()
    rt.join(timeout=90)
    wall = time.time() - t0
    assert recv_done.is_set(), "90s 内未收到最后一帧结果(链路卡死)"
    print(f"A 灌帧完成: {N_PUSH} 帧, {len(results)} 个结果, 耗时 {wall:.1f}s", flush=True)

    # 帧号协议: 结果帧号严格递增, 且最后一帧(最新)必然存活
    idxs = [ev["frame_idx"] for ev in results if "frame_idx" in ev]
    assert idxs == sorted(idxs) and len(set(idxs)) == len(idxs), f"结果帧号乱序/重复: {idxs[:20]}"
    assert idxs[-1] == N_PUSH - 1, f"最新帧必须存活(合并缓冲的底线): {idxs[-1]}"

    # 丢弃账恒等式: j_k - j_{k-1} - 1 == 该结果携带的 transport_dropped
    total_dropped, prev = 0, -1
    for ev in results:
        j = ev["frame_idx"]
        td = ev.get("flow", {}).get("transport_dropped", 0)
        assert j - prev - 1 == td, \
            f"丢弃账不平: idx={j}, prev={prev}, 上报 dropped={td}"
        total_dropped += td
        prev = j
    assert len(results) + total_dropped == N_PUSH, \
        f"总账不平: {len(results)} 结果 + {total_dropped} 丢弃 != {N_PUSH}"
    assert total_dropped > 0, "25fps 灌帧应有过载丢弃(若 0 丢弃说明处理跟得上, 也需记录)"
    print(f"PASS A1 丢弃账恒等式: {len(results)} 处理 + {total_dropped} 传输丢弃 = {N_PUSH}", flush=True)

    # flow 观测: 每个结果都有流控数据
    flows = [ev["flow"] for ev in results]
    assert all("ema_arrival_ms" in f and "stride" in f for f in flows)
    assert all(f["stride"] >= 1 for f in flows), "stride 不得低于配置"
    proc = [f["ema_proc_ms"] for f in flows if f.get("ema_proc_ms")]
    assert proc, "追踪帧应有处理耗时 EMA"
    print(f"PASS A2 flow 观测: proc≈{proc[-1]:.0f}ms, stride={flows[-1]['stride']}", flush=True)

    # 空洞帧(被丢弃)加提示 -> 拒绝; 最新帧(已入库)加提示 -> 真实掩码
    holes = sorted(set(range(N_PUSH)) - set(idxs))
    assert holes, "应有丢弃产生的空洞帧"
    try:
        ch.add_point(group_id=1, x=480, y=800, label=1, frame_idx=holes[0])
        raise AssertionError("空洞帧加提示应抛 ApiError")
    except ApiError as e:
        assert "未入库" in str(e), f"错误文案不对: {e}"
    print(f"PASS A3 空洞帧 {holes[0]} 提示被拒绝(共 {len(holes)} 个空洞)", flush=True)

    latest = idxs[-1]
    t0 = time.time()
    ev = ch.add_point(group_id=1, x=480, y=800, label=1, frame_idx=latest)
    assert ev["computed"] and ev["groups"] == [1]
    assert set(ev["mask_images"]) == {1} and ev["mask_images"][1].size == frames[0].size
    print(f"PASS A4 已入库帧 {latest} 提示 -> 真实掩码 ({time.time() - t0:.2f}s)", flush=True)

    # submit 计划只遍历已入库帧(空洞不在计划内, 不炸)
    kf = [e.get("frame_idx") for e in ch.submit(start_frame=0) if e["type"] == "keyframe"]
    assert set(kf) <= set(idxs) | {latest}, f"计划含未入库帧: {kf}"
    assert holes[0] not in kf, "空洞帧不应出现在传播计划里"
    print(f"PASS A5 submit 计划容忍空洞: {len(kf)} 个关键帧", flush=True)
    ch.close()
    client.close_video_session(sid)

    # ================= 场景B: 旧协议(无帧号头)兼容 =================
    sid2 = client.create_stream_session(frame_stride=1, auto_predict=True)
    ch2 = client.open_video(sid2)
    evs = [ch2.push_frame(frames[i]) for i in range(3)]   # 阻塞式, 不带 frame_idx
    assert [e["frame_idx"] for e in evs] == [0, 1, 2], \
        f"旧协议应按到达顺序编号: {[e['frame_idx'] for e in evs]}"
    assert all(e["keyframe"] for e in evs)
    ch2.close()
    client.close_video_session(sid2)
    print("PASS B 旧协议(裸图像字节)兼容, 顺序编号正确", flush=True)

    server.should_exit = True
    print("\nSTREAM FLOW CONTROL SMOKE OK", flush=True)


if __name__ == "__main__":
    main()

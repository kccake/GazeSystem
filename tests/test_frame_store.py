import glob
import os

import torch


from GazeSystem.compute.frame_store import DiskFrameStore


def test_roundtrip(tmp_path):
    store = DiskFrameStore(store_dir=str(tmp_path))
    frames = [torch.randn(3, 1008, 1008, dtype=torch.bfloat16) for _ in range(4)]
    for i, f in enumerate(frames):
        store[i] = f
    assert len(store) == 4
    # flush 前: 从 buffer 读
    assert torch.equal(store[2], frames[2])
    store.flush()
    # flush 后: 从段文件读(mmap)
    for i, f in enumerate(frames):
        out = store[i]
        assert out.dtype == torch.bfloat16
        assert torch.equal(out, f)
    assert len(glob.glob(os.path.join(str(tmp_path), "*.safetensors"))) == 1
    store.close()
    assert glob.glob(os.path.join(str(tmp_path), "*.safetensors")) == []


def test_out_of_order_write_rejected(tmp_path):
    store = DiskFrameStore(store_dir=str(tmp_path))
    store[0] = torch.randn(3, 1008, 1008, dtype=torch.bfloat16)
    try:
        store[5] = torch.randn(3, 1008, 1008, dtype=torch.bfloat16)
        assert False, "乱序写入应该被拒绝"
    except IndexError:
        pass
    store.close()

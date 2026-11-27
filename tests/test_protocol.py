"""
推帧消息协议测试(帧号头 + 旧协议魔数嗅探兼容)
"""
import sys
from pathlib import Path
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # codeRepo 根
from GazeSystem.api.protocol import (encode_image, pack_push_frame,
                                     unpack_push_frame)


def test_push_frame_roundtrip():
    """帧号头打包/解包往返一致"""
    payload = encode_image(Image.new("RGB", (8, 8)), "JPEG")
    idx, out = unpack_push_frame(pack_push_frame(42, payload))
    assert idx == 42 and out == payload
    print("PASS test_push_frame_roundtrip")


def test_push_frame_legacy_sniff():
    """旧协议(裸 JPEG/PNG 字节)被嗅探为无头, 原样返回"""
    for fmt in ("JPEG", "PNG"):
        payload = encode_image(Image.new("RGB", (8, 8)), fmt)
        idx, out = unpack_push_frame(payload)
        assert idx is None and out == payload, f"{fmt} 旧协议应原样返回"
    print("PASS test_push_frame_legacy_sniff")


def test_push_frame_no_magic_collision():
    """帧号头不会被误判为裸图像(魔数冲突需 idx >= 0xFFD80000, 约43亿帧)"""
    payload = encode_image(Image.new("RGB", (8, 8)), "JPEG")  # FF D8 开头
    for i in (0, 1, 65535, 65536, 10 ** 6):
        idx, out = unpack_push_frame(pack_push_frame(i, payload))
        assert idx == i and out == payload, f"idx={i} 不应被误判为裸 JPEG"
    print("PASS test_push_frame_no_magic_collision")


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

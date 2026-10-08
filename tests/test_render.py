import numpy as np

from src.framing.render import crop_region, warp_to_output


def test_full_frame_crop_is_identity():
    rng = np.random.default_rng(0)
    frame = rng.integers(0, 256, (360, 640, 3), dtype=np.uint8)
    crop = np.array([0.0, 0.0, 640.0, 360.0])
    region, offset = crop_region(frame, crop)
    out = warp_to_output(region, crop, offset, (640, 360), "linear")
    assert np.array_equal(out, frame)


def test_subpixel_pan_moves_smoothly():
    """慢速平移 0.25px / 幀：輸出應連續變化，不會每 4 幀才跳 1px。"""
    x = np.arange(640, dtype=np.float32)
    row = (128 + 100 * np.sin(x / 20)).astype(np.uint8)
    frame = np.ascontiguousarray(np.broadcast_to(row[None, :, None], (360, 640, 3)))
    prev = None
    diffs = []
    for i in range(8):
        crop = np.array([100 + 0.25 * i, 50.0, 420 + 0.25 * i, 230.0])
        region, offset = crop_region(frame, crop)
        out = warp_to_output(region, crop, offset, (640, 360), "linear").astype(np.float32)
        if prev is not None:
            diffs.append(np.abs(out - prev).mean())
        prev = out
    assert min(diffs) > 0                                   # 每一幀都有移動


def test_large_downscale_has_expected_size():
    frame = np.full((2160, 3840, 3), 128, np.uint8)
    crop = np.array([0.0, 0.0, 3840.0, 2160.0])
    region, offset = crop_region(frame, crop)
    out = warp_to_output(region, crop, offset, (960, 540))
    assert out.shape == (540, 960, 3)
    assert np.all(np.abs(out.astype(int) - 128) <= 1)

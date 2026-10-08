import cv2
import numpy as np

from src.lighting.masks import face_mask, face_region
from src.lighting.restore import FaceRestorer, highlight_ratio, restore_roi, to_lab

from .helpers import make_person


def synthetic_scene(face_value: int = 252):
    """暗色舞台背景 + 一張過亮的臉（膚色、帶一點紋理）。"""
    rng = np.random.default_rng(1)
    img = np.zeros((720, 1280, 3), np.uint8)
    img[:] = (40, 30, 60)
    img[:, :, 2] = np.linspace(40, 120, 1280, dtype=np.uint8)[None, :]   # 背景漸層
    person = make_person(cx=640, top=200, height=420, track_id=1)
    face = face_region(person)
    skin = np.array([face_value - 30, face_value - 12, face_value], np.float32)   # BGR 膚色
    canvas = np.zeros(img.shape[:2], np.float32)
    cv2.ellipse(canvas, ((face.cx, face.cy), (face.width * 0.8, face.height * 0.8), 0), 1.0, -1)
    texture = rng.normal(0, 4, img.shape).astype(np.float32)
    face_px = np.clip(skin + texture, 0, 255)
    sel = canvas > 0
    img[sel] = face_px[sel].astype(np.uint8)
    return img, person, face


def test_overexposed_ratio_drops(lighting_cfg):
    img, person, face = synthetic_scene()
    mask, (x0, y0, x1, y1) = face_mask(face, img.shape)
    roi = img[y0:y1, x0:x1]
    before = highlight_ratio(to_lab(roi)[..., 0] / 100, mask, lighting_cfg.highlight_thr)
    fixed = restore_roi(roi, mask, 1.0, lighting_cfg)
    after = highlight_ratio(to_lab(fixed)[..., 0] / 100, mask, lighting_cfg.highlight_thr)
    assert before > 0.3
    assert after < before * 0.5


def test_pixels_outside_mask_unchanged(lighting_cfg):
    img, person, face = synthetic_scene()
    lighting_cfg.strength_ema = 1.0                        # 不做時間平滑，第一幀就全強度
    out = img.copy()
    stats = FaceRestorer(lighting_cfg).apply(out, [person])
    assert stats and stats[0].strength > 0.9

    mask, (x0, y0, x1, y1) = face_mask(face, img.shape)
    outside = np.ones(img.shape[:2], bool)
    outside[y0:y1, x0:x1] = mask <= 0
    assert np.array_equal(out[outside], img[outside])     # 不動布景
    assert not np.array_equal(out, img)


def test_chroma_unchanged(lighting_cfg):
    img, person, face = synthetic_scene(face_value=235)
    mask, (x0, y0, x1, y1) = face_mask(face, img.shape)
    roi = img[y0:y1, x0:x1]
    fixed = restore_roi(roi, mask, 1.0, lighting_cfg)
    lab_in = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB).astype(np.int16)
    lab_out = cv2.cvtColor(fixed, cv2.COLOR_BGR2LAB).astype(np.int16)
    sel = mask > 0.5
    assert np.abs(lab_out[..., 0][sel] - lab_in[..., 0][sel]).mean() > 3   # 亮度確實有變
    da = np.abs(lab_out[..., 1:][sel] - lab_in[..., 1:][sel])
    assert da.mean() < 1.0 and da.max() <= 3                              # 不改膚色（只剩量化誤差）


def test_well_exposed_face_untouched(lighting_cfg):
    img, person, _ = synthetic_scene(face_value=170)
    lighting_cfg.strength_ema = 1.0
    out = img.copy()
    stats = FaceRestorer(lighting_cfg).apply(out, [person])
    assert stats[0].strength == 0.0
    assert np.array_equal(out, img)


def test_strength_is_smoothed_per_track(lighting_cfg):
    img, person, _ = synthetic_scene()
    restorer = FaceRestorer(lighting_cfg)
    strengths = [restorer.apply(img.copy(), [person])[0].strength for _ in range(10)]
    assert strengths[0] < 0.2                              # 新出現的人從 0 淡入
    assert all(b > a for a, b in zip(strengths, strengths[1:]))


def test_face_region_requires_head_keypoints():
    person = make_person(640, 200, 420)
    person.scores[:3] = 0.0                                # 鼻子、兩眼都不可見（背對）
    assert face_region(person) is None


def test_mask_with_offset_matches_full_frame(lighting_cfg):
    img, person, face = synthetic_scene()
    full, (fx0, fy0, fx1, fy1) = face_mask(face, img.shape)
    ox, oy = 500, 100                                      # 臉完整落在子區域內
    sub, (sx0, sy0, sx1, sy1) = face_mask(face, img[oy:, ox:].shape, offset=(ox, oy))
    assert (sx0 + ox, sy0 + oy, sx1 + ox, sy1 + oy) == (fx0, fy0, fx1, fy1)
    assert np.array_equal(sub, full)

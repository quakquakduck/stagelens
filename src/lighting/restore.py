"""臉部燈光還原：只壓高光、恢復局部對比、不改形狀、不改膚色。

- 只處理 LAB 的 L 通道，a/b 色度不動 → 膚色、妝容顏色不變。
- 只在臉部遮罩內混合；遮罩外的像素完全不變（不動布景）。
- 已經過曝到 255 的像素沒有細節可以還原，只能讓它不那麼刺眼並恢復周圍的漸層。
"""
from __future__ import annotations

import cv2
import numpy as np

from ..datatypes import FaceStat, Person
from .masks import face_mask, face_region


def to_lab(roi_bgr: np.ndarray) -> np.ndarray:
    """uint8 BGR → float32 LAB（L ∈ [0, 100]）。用浮點避免 uint8 來回轉換的量化誤差。"""
    return cv2.cvtColor(roi_bgr.astype(np.float32) * (1.0 / 255.0), cv2.COLOR_BGR2LAB)


def highlight_ratio(L: np.ndarray, mask: np.ndarray, thr: float) -> float:
    """臉部（mask > 0.5）內 L > thr 的像素比例；L 為 0–1。"""
    sel = mask > 0.5
    n = int(sel.sum())
    return float((L[sel] > thr).sum()) / n if n else 0.0


def strength_from_ratio(ratio: float) -> float:
    """沒過曝就不動：r ≤ 2% → 0，r ≥ 17% → 1。"""
    return float(np.clip((ratio - 0.02) / 0.15, 0.0, 1.0))


def restore_roi(roi_bgr: np.ndarray, mask: np.ndarray, strength: float, cfg,
                lab: np.ndarray | None = None) -> np.ndarray:
    """roi_bgr: uint8 ROI；mask: float32 0–1，同尺寸。回傳新的 uint8 ROI。"""
    if lab is None:
        lab = to_lab(roi_bgr)
    else:
        lab = lab.copy()
    L = lab[..., 0] * 0.01

    # 高光 soft-knee 壓縮：L > knee 的部分以斜率 compress 壓下來
    k, s = cfg.knee, cfg.compress
    Lc = np.where(L > k, k + (L - k) * s, L)

    # CLAHE 恢復局部對比。只取 CLAHE 造成的「高頻」變化 × detail：直接用 CLAHE 結果會把剛壓下來的
    # 高光重新拉伸回接近全白（實測臉部過曝比例 0% → 17%），低頻亮度必須沿用壓縮後的 Lc。
    tiles = int(cfg.clahe_tiles)
    clahe = cv2.createCLAHE(clipLimit=cfg.clahe_clip, tileGridSize=(tiles, tiles))
    Lclahe = clahe.apply(np.round(Lc * 255).astype(np.uint8)).astype(np.float32) / 255.0
    delta = Lclahe - Lc
    sigma = max(1.0, min(L.shape) / (2 * tiles))           # 約半個 CLAHE tile
    Lfix = Lc + cfg.detail * (delta - cv2.GaussianBlur(delta, (0, 0), sigma))

    a = mask * strength
    lab[..., 0] = (L + (Lfix - L) * a) * 100.0
    out = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    out = np.clip(out * 255.0 + 0.5, 0, 255).astype(np.uint8)
    untouched = a <= 0
    out[untouched] = roi_bgr[untouched]
    return out


class FaceRestorer:
    """對每個人的臉做局部燈光還原；修正強度依 track_id 做時間平滑，避免畫面閃爍。"""

    def __init__(self, cfg, collect_stats: bool = False, forget_after: int = 90):
        self.cfg = cfg
        self.collect_stats = collect_stats
        self.forget_after = forget_after
        self._strength: dict[int, float] = {}
        self._last_seen: dict[int, int] = {}
        self._frame = 0

    def apply(self, image: np.ndarray, persons: list[Person],
              offset: tuple[int, int] = (0, 0)) -> list[FaceStat]:
        """就地修改 image（左上角位於原畫面座標 offset，例如裁切區）。"""
        c = self.cfg
        stats = []
        for p in persons:
            face = face_region(p, c.feather, c.min_face_px)
            if face is None:
                continue
            got = face_mask(face, image.shape, offset)
            if got is None:
                continue
            mask, (x0, y0, x1, y1) = got
            roi = image[y0:y1, x0:x1]
            lab = to_lab(roi)
            ratio = highlight_ratio(lab[..., 0] * 0.01, mask, c.highlight_thr)
            strength = self._smooth(p.track_id, strength_from_ratio(ratio))

            after = ratio if self.collect_stats else None
            if strength > 1e-3:
                fixed = restore_roi(roi, mask, strength, c, lab=lab)
                image[y0:y1, x0:x1] = fixed
                if self.collect_stats:
                    after = highlight_ratio(to_lab(fixed)[..., 0] * 0.01, mask, c.highlight_thr)
            stats.append(FaceStat(p.track_id, strength, ratio, after))

        self._frame += 1
        stale = [t for t, f in self._last_seen.items() if self._frame - f > self.forget_after]
        for t in stale:
            self._strength.pop(t, None)
            self._last_seen.pop(t, None)
        return stats

    def _smooth(self, track_id: int, strength: float) -> float:
        if track_id < 0:              # 沒有追蹤 ID 時無法做時間平滑
            return strength
        prev = self._strength.get(track_id, 0.0)   # 新出現的人從 0 慢慢淡入
        value = prev + self.cfg.strength_ema * (strength - prev)
        self._strength[track_id] = value
        self._last_seen[track_id] = self._frame
        return value

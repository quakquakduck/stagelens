"""運鏡 / 燈光量化指標（寫進 docs/benchmarks.md）。

| 指標         | 算法                                                     |
|--------------|----------------------------------------------------------|
| 主體完整率   | 有效人物的可見關鍵點全部落在裁切框內的幀數比例           |
| 平順度       | 裁切中心二階差分（加速度）的平均絕對值，越低越穩         |
| 平均放大倍率 | W / 裁切寬度，證明「動作不再看起來很小」                 |
| 臉部過曝比例 | 臉部 ROI 中 L > highlight_thr 的像素比例，修正前 vs 後   |
"""
from __future__ import annotations

import numpy as np

from ..datatypes import FrameResult
from ..framing.target import valid_persons


class QualityMetrics:
    def __init__(self, W: int, H: int, framing_cfg):
        self.W, self.H, self.cfg = W, H, framing_cfg
        self.centers: list[tuple[float, float]] = []
        self.zooms: list[float] = []
        self.subject_frames = 0
        self.complete_frames = 0
        self.face_before: list[float] = []
        self.face_after: list[float] = []

    def update(self, res: FrameResult) -> None:
        if res.crop is None:
            return
        x0, y0, x1, y1 = res.crop
        self.centers.append(((x0 + x1) / 2, (y0 + y1) / 2))
        self.zooms.append(self.W / (x1 - x0))

        valid = valid_persons(res.persons, self.H, self.cfg)
        if valid:
            self.subject_frames += 1
            pts = np.concatenate([p.keypoints[p.visible()] for p in valid])
            inside = ((pts[:, 0] >= x0) & (pts[:, 0] <= x1) & (pts[:, 1] >= y0) & (pts[:, 1] <= y1)).all()
            self.complete_frames += int(inside)

        for f in res.faces:
            if f.over_after is not None:
                self.face_before.append(f.over_before)
                self.face_after.append(f.over_after)

    def summary(self) -> dict[str, float | None]:
        c = np.array(self.centers, dtype=np.float64).reshape(-1, 2)
        jerk = np.linalg.norm(np.diff(c, n=2, axis=0), axis=1) if len(c) >= 3 else np.zeros(0)
        return {
            "subject_complete_ratio": self.complete_frames / self.subject_frames if self.subject_frames else None,
            "subject_frames": self.subject_frames,
            "smoothness_px": float(jerk.mean()) if len(jerk) else None,
            "smoothness_pct_w": float(jerk.mean() / self.W * 100) if len(jerk) else None,
            "mean_zoom": float(np.mean(self.zooms)) if self.zooms else None,
            "face_samples": len(self.face_before),
            "face_overexposed_before": float(np.mean(self.face_before)) if self.face_before else None,
            "face_overexposed_after": float(np.mean(self.face_after)) if self.face_after else None,
        }

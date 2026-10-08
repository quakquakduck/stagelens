"""模組之間傳遞的資料結構。

原則：模組之間只傳遞這些 dataclass，不傳遞模型物件。Stage II 換成 Hailo 後端時，
framing / lighting / overlay / output 一行都不用改。

（實作指引中此檔名為 types.py；改名是為了避免以 `python src/main.py` 執行時
遮蔽標準庫的 `types` 模組。）
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# COCO-17 關鍵點索引
NOSE, L_EYE, R_EYE, L_EAR, R_EAR = 0, 1, 2, 3, 4
L_SHOULDER, R_SHOULDER = 5, 6
L_ELBOW, R_ELBOW = 7, 8
L_WRIST, R_WRIST = 9, 10
L_HIP, R_HIP = 11, 12
L_KNEE, R_KNEE = 13, 14
L_ANKLE, R_ANKLE = 15, 16
NUM_KEYPOINTS = 17

KPT_THR = 0.3        # 關鍵點視為可見的分數門檻
MIN_KEYPOINTS = 5    # 可見關鍵點少於此數視為誤偵測


@dataclass
class Person:
    keypoints: np.ndarray          # (17, 2) float32，原始畫面像素座標
    scores: np.ndarray             # (17,)  float32，0–1
    bbox: np.ndarray               # (4,)   x0, y0, x1, y1
    track_id: int = -1

    def visible(self, thr: float = KPT_THR) -> np.ndarray:
        return self.scores >= thr


@dataclass
class FaceStat:
    track_id: int
    strength: float                # 實際套用的修正強度（時間平滑後）
    over_before: float             # 臉部 L > highlight_thr 的像素比例（修正前）
    over_after: float | None = None  # 修正後（只在 collect_stats 時計算）


@dataclass
class FrameResult:
    index: int
    timestamp: float               # 秒
    persons: list[Person] = field(default_factory=list)
    crop: np.ndarray | None = None    # (4,) x0, y0, x1, y1（float）
    target: np.ndarray | None = None  # (3,) cx, cy, crop_w；沒人時為 None
    faces: list[FaceStat] = field(default_factory=list)
    timings: dict[str, float] = field(default_factory=dict)  # ms

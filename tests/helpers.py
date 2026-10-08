from __future__ import annotations

import numpy as np

from src.datatypes import Person

W, H = 1920, 1080

def make_person(cx: float, top: float, height: float, track_id: int = -1, score: float = 0.9) -> Person:
    """手造一個正面站立的人：頭頂（眼睛）在 top，腳踝在 top + height，水平中心 cx。"""
    u = height / 7.0                     # 粗略人體比例單位
    rel = np.array([
        (0.0, 0.25),                     # nose
        (0.15, 0.0), (-0.15, 0.0),       # eyes（人的左眼在畫面右側）
        (0.35, 0.1), (-0.35, 0.1),       # ears
        (0.9, 1.2), (-0.9, 1.2),         # shoulders
        (1.1, 2.4), (-1.1, 2.4),         # elbows
        (1.2, 3.5), (-1.2, 3.5),         # wrists
        (0.5, 3.6), (-0.5, 3.6),         # hips
        (0.5, 5.3), (-0.5, 5.3),         # knees
        (0.5, 7.0), (-0.5, 7.0),         # ankles
    ], dtype=np.float32)
    kpts = np.stack([cx + rel[:, 0] * u, top + rel[:, 1] * u], axis=1).astype(np.float32)
    scores = np.full(17, score, dtype=np.float32)
    bbox = np.array([*kpts.min(0), *kpts.max(0)], dtype=np.float32)
    return Person(kpts, scores, bbox, track_id)

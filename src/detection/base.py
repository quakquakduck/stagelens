from __future__ import annotations

from abc import ABC, abstractmethod

import cv2
import numpy as np

from ..datatypes import KPT_THR, MIN_KEYPOINTS, Person


class PoseBackend(ABC):
    name = "pose"

    @abstractmethod
    def infer(self, frame_bgr: np.ndarray) -> list[Person]:
        """輸入 BGR 原始畫面，回傳畫面座標系下的 Person 清單。"""

    def close(self) -> None:
        """釋放模型資源（Hailo 的 VDevice 等）。"""


class ScaledBackend(PoseBackend):
    """推論前把畫面長邊縮到 long_side，再把座標乘回原比例。適用任何後端。"""

    def __init__(self, inner: PoseBackend, long_side: int):
        self.inner = inner
        self.long_side = long_side

    @property
    def name(self) -> str:
        return f"{self.inner.name} (long side {self.long_side})"

    def infer(self, frame_bgr: np.ndarray) -> list[Person]:
        h, w = frame_bgr.shape[:2]
        s = self.long_side / max(h, w)
        if s >= 1.0:
            return self.inner.infer(frame_bgr)
        small = cv2.resize(frame_bgr, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
        sx, sy = w / small.shape[1], h / small.shape[0]
        persons = self.inner.infer(small)
        scale = np.array([sx, sy], dtype=np.float32)
        for p in persons:
            p.keypoints = p.keypoints * scale
            p.bbox = p.bbox * np.tile(scale, 2)
        return persons

    def close(self) -> None:
        self.inner.close()


def persons_from_arrays(keypoints: np.ndarray, scores: np.ndarray,
                        kpt_thr: float = KPT_THR, min_keypoints: int = MIN_KEYPOINTS) -> list[Person]:
    """(N,17,2) 關鍵點 + (N,17) 分數 → Person 清單；bbox 取可見關鍵點的外框。

    所有後端（rtmlib / Hailo）共用，確保輸出格式一致。
    """
    persons = []
    for k, s in zip(np.asarray(keypoints), np.asarray(scores)):
        vis = s >= kpt_thr
        if vis.sum() < min_keypoints:          # 關鍵點太少視為誤偵測
            continue
        p = k[vis]
        bbox = np.array([*p.min(0), *p.max(0)], dtype=np.float32)
        persons.append(Person(k.astype(np.float32), s.astype(np.float32), bbox))
    return persons

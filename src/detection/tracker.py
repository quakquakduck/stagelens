"""簡易 IoU 追蹤：貪婪配對，給每個人穩定的 track_id。

ID 用來讓每個人的燈光修正強度各自做時間平滑，也是 Stage II 主角選擇的基礎。
"""
from __future__ import annotations

import numpy as np

from ..datatypes import Person


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """a: (N,4)、b: (M,4) xyxy → (N,M) IoU。"""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    x0 = np.maximum(a[:, None, 0], b[None, :, 0])
    y0 = np.maximum(a[:, None, 1], b[None, :, 1])
    x1 = np.minimum(a[:, None, 2], b[None, :, 2])
    y1 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / np.maximum(union, 1e-6)


class Tracker:
    def __init__(self, iou_thr: float = 0.3, max_age: int = 15):
        self.iou_thr = iou_thr
        self.max_age = max_age        # 連續幾次 update 沒配對到就刪除
        self._next_id = 0
        self._tracks: dict[int, tuple[np.ndarray, int]] = {}   # id → (bbox, 未配對次數)

    def update(self, persons: list[Person]) -> list[Person]:
        ids = list(self._tracks)
        boxes = np.array([self._tracks[i][0] for i in ids], dtype=np.float32).reshape(-1, 4)
        dets = np.array([p.bbox for p in persons], dtype=np.float32).reshape(-1, 4)
        iou = iou_matrix(boxes, dets)

        matched_t, matched_d = set(), set()
        for flat in np.argsort(-iou, axis=None):
            t, d = np.unravel_index(flat, iou.shape)
            if iou[t, d] < self.iou_thr:
                break
            if t in matched_t or d in matched_d:
                continue
            matched_t.add(t)
            matched_d.add(d)
            persons[d].track_id = ids[t]

        tracks: dict[int, tuple[np.ndarray, int]] = {}
        for t, tid in enumerate(ids):
            if t not in matched_t:
                box, misses = self._tracks[tid]
                if misses + 1 <= self.max_age:
                    tracks[tid] = (box, misses + 1)
        for d, p in enumerate(persons):
            if d not in matched_d:
                p.track_id = self._next_id
                self._next_id += 1
            tracks[p.track_id] = (p.bbox.copy(), 0)
        self._tracks = tracks
        return persons

    def reset(self) -> None:
        self._tracks.clear()

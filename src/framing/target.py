"""(a) 由關鍵點算出這一幀「理想」的裁切目標 (cx, cy, crop_w)。"""
from __future__ import annotations

from collections import deque

import numpy as np

from ..datatypes import MIN_KEYPOINTS, Person


def valid_persons(persons: list[Person], H: int, cfg) -> list[Person]:
    """可見關鍵點 ≥ 5 且 bbox 高度 ≥ 畫面高度 min_person_h 的人。"""
    return [p for p in persons
            if p.visible().sum() >= MIN_KEYPOINTS and (p.bbox[3] - p.bbox[1]) >= cfg.min_person_h * H]


def subject_box(persons: list[Person], W: int, H: int, cfg) -> np.ndarray | None:
    """所有有效人物可見關鍵點的外框 + 留白；人太分散時回傳全景。沒人時回傳 None。"""
    valid = valid_persons(persons, H, cfg)
    if not valid:
        return None
    pts = np.concatenate([p.keypoints[p.visible()] for p in valid])
    x0, y0 = pts.min(0)
    x1, y1 = pts.max(0)
    w, h = x1 - x0, y1 - y0
    if w > cfg.spread_full * W:
        return np.array([0.0, 0.0, W, H])
    # 頭頂要多留：關鍵點最高只到眼睛 / 耳朵，不加會切頭
    return np.array([x0 - cfg.pad_side * w, y0 - cfg.pad_top * h,
                     x1 + cfg.pad_side * w, y1 + cfg.pad_bottom * h], dtype=np.float64)


def box_to_target(box: np.ndarray, W: int, H: int, cfg) -> np.ndarray:
    """外框 → 符合輸出比例、縮放受限的 (cx, cy, crop_w)。"""
    x0, y0, x1, y1 = box
    crop_w = max(x1 - x0, (y1 - y0) * cfg.aspect)
    crop_w = float(np.clip(crop_w, W / cfg.zoom_max, min(W, H * cfg.aspect)))
    return np.array([(x0 + x1) / 2, (y0 + y1) / 2, crop_w])


def compute_target(persons: list[Person], W: int, H: int, cfg) -> np.ndarray | None:
    """回傳 (cx, cy, crop_w)；沒人時回傳 None。不含時間窗，見 TargetPlanner。"""
    box = subject_box(persons, W, H, cfg)
    return None if box is None else box_to_target(box, W, H, cfg)


def target_to_rect(target: np.ndarray, aspect: float) -> np.ndarray:
    cx, cy, cw = target
    ch = cw / aspect
    return np.array([cx - cw / 2, cy - ch / 2, cx + cw / 2, cy + ch / 2])


class TargetPlanner:
    """大動作放寬：取最近 burst_window_s 秒內所有目標框的聯集。

    跳躍、伸展手腳時畫面立刻放大，但動作結束後不會馬上縮回去；
    偵測偶爾漏掉一兩次時也會沿用時間窗內的框，不會觸發「跟丟」。
    """

    def __init__(self, W: int, H: int, fps: float, cfg):
        self.W, self.H, self.cfg = W, H, cfg
        self.window = max(1, round(cfg.burst_window_s * fps))
        self._history: deque[tuple[int, np.ndarray]] = deque()
        self._frame = 0

    def update(self, persons: list[Person]) -> np.ndarray | None:
        box = subject_box(persons, self.W, self.H, self.cfg)
        if box is not None:
            self._history.append((self._frame, box))
        while self._history and self._history[0][0] <= self._frame - self.window:
            self._history.popleft()
        self._frame += 1
        if not self._history:
            return None
        boxes = np.array([b for _, b in self._history])
        union = np.array([boxes[:, 0].min(), boxes[:, 1].min(), boxes[:, 2].max(), boxes[:, 3].max()])
        return box_to_target(union, self.W, self.H, self.cfg)

    def reset(self) -> None:
        self._history.clear()

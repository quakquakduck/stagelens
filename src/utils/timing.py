"""各階段計時：`with timer("detect"): ...`，每幀結束呼叫 end_frame()。"""
from __future__ import annotations

import csv
import time
from collections import deque
from contextlib import contextmanager
from pathlib import Path

import numpy as np


class StageTimer:
    def __init__(self, fps_window: int = 30):
        self.rows: list[dict[str, float]] = []
        self._cur: dict[str, float] = {}
        self._frame_ends: deque[float] = deque(maxlen=fps_window)

    @contextmanager
    def __call__(self, name: str):
        t = time.perf_counter()
        try:
            yield
        finally:
            self.add(name, (time.perf_counter() - t) * 1000.0)

    def add(self, name: str, ms: float) -> None:
        self._cur[name] = self._cur.get(name, 0.0) + ms

    def end_frame(self, **extra: float) -> dict[str, float]:
        row = {**extra, **self._cur}
        self.rows.append(row)
        self._cur = {}
        self._frame_ends.append(time.perf_counter())
        return row

    @property
    def last(self) -> dict[str, float]:
        return self.rows[-1] if self.rows else {}

    def fps(self) -> float:
        if len(self._frame_ends) < 2:
            return 0.0
        dt = self._frame_ends[-1] - self._frame_ends[0]
        return (len(self._frame_ends) - 1) / dt if dt > 0 else 0.0

    def stage_names(self) -> list[str]:
        names: list[str] = []
        for row in self.rows:
            names += [k for k in row if k not in names]
        return names

    def summary(self, skip: int = 0) -> dict[str, dict[str, float]]:
        """每個欄位的 mean / median / p95（ms）；skip 跳過前幾幀（模型暖機）。"""
        rows = self.rows[skip:]
        out = {}
        for name in self.stage_names():
            vals = np.array([r[name] for r in rows if name in r], dtype=np.float64)
            if len(vals):
                out[name] = {"mean": float(vals.mean()), "median": float(np.median(vals)),
                             "p95": float(np.percentile(vals, 95)), "n": int(len(vals))}
        return out

    def dump_csv(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        names = self.stage_names()
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=names)
            w.writeheader()
            for row in self.rows:
                w.writerow({k: (f"{v:.3f}" if isinstance(v, float) else v) for k, v in row.items()})

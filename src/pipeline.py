"""單幀處理流程：偵測 → 運鏡 → 燈光 → 放大 → 組合畫面。main.py 與 scripts/benchmark.py 共用。"""
from __future__ import annotations

import numpy as np

from .datatypes import FrameResult, Person
from .detection import PoseBackend, Tracker, make_backend
from .framing import TargetPlanner, VirtualCamera, crop_region, target_to_rect, warp_to_output
from .lighting import FaceRestorer
from .overlay import compose_view, render_size
from .utils import StageTimer


class Pipeline:
    def __init__(self, cfg, width: int, height: int, fps: float,
                 backend: PoseBackend | None = None, collect_stats: bool = False):
        self.cfg = cfg
        self.W, self.H = width, height
        self.backend = backend if backend is not None else make_backend(cfg)
        self.tracker = Tracker(cfg.tracker.iou_thr, cfg.tracker.max_age)
        self.planner = TargetPlanner(width, height, fps, cfg.framing)
        self.camera = VirtualCamera(width, height, cfg.framing)
        self.restorer = FaceRestorer(cfg.lighting, collect_stats) if cfg.lighting.enabled else None
        self.timer = StageTimer()
        self.debug = cfg.debug or cfg.view == "debug"
        self.render_size = render_size(cfg.view, cfg.output_size)
        self.persons: list[Person] = []

    def process(self, idx: int, frame: np.ndarray, timestamp: float = 0.0) -> tuple[np.ndarray, FrameResult]:
        t = self.timer
        if idx % self.cfg.detect_every == 0:      # 隔幀推論：中間幀沿用上次結果
            with t("detect"):
                self.persons = self.tracker.update(self.backend.infer(frame))
        persons = self.persons

        with t("framing"):
            target = self.planner.update(persons)
            crop = self.camera.update(target)

        faces = []
        region, offset = crop_region(frame, crop)
        if self.restorer is not None:
            with t("lighting"):               # 先裁切 → 在原解析度的裁切區修正 → 再放大
                region = region.copy()
                faces = self.restorer.apply(region, persons, offset)

        with t("render"):
            out = warp_to_output(region, crop, offset, self.render_size, self.cfg.framing.interp)

        with t("compose"):
            target_rect = target_to_rect(target, self.cfg.framing.aspect) if target is not None else None
            hud = self.hud_lines(len(persons)) if self.debug else None
            view = compose_view(self.cfg.view, frame, out, self.cfg.output_size,
                                persons, crop, target_rect, hud, self.debug)

        return view, FrameResult(idx, timestamp, persons, crop, target, faces)

    def hud_lines(self, n_persons: int) -> list[str]:
        last = self.timer.last
        stages = "  ".join(f"{k} {last[k]:.1f}" for k in ("detect", "framing", "lighting", "render", "compose", "write")
                           if k in last)
        return [
            f"FPS {self.timer.fps():.1f}   latency {last.get('latency', 0.0):.0f} ms",
            f"{stages} ms" if stages else "",
            f"zoom {self.camera.zoom:.2f}x   persons {n_persons}   {getattr(self.backend, 'name', '')}",
        ]

    def close(self) -> None:
        self.backend.close()

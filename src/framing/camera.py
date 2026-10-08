"""(b) 虛擬攝影機：讓裁切框「平滑地追」目標。

| 機制               | 作用                                   |
|--------------------|----------------------------------------|
| Dead-zone          | 目標只移動一點點時完全不動，消除抖動   |
| EMA / 比例追蹤     | 速度 ∝ 誤差，越接近越慢                |
| 速度 / 加速度上限  | 不會突然甩鏡頭，起步與停止都有緩衝     |
| 放大慢、縮小快     | 寧可畫面寬一點，也不要切到手腳         |
"""
from __future__ import annotations

import numpy as np


class VirtualCamera:
    def __init__(self, W: int, H: int, cfg):
        self.W, self.H, self.cfg = W, H, cfg
        self.reset()

    def reset(self) -> None:
        self.state = np.array([self.W / 2, self.H / 2, self.W], dtype=np.float64)  # cx, cy, crop_w
        self.vel = np.zeros(3)
        self.lost = 0
        self._clamp()

    def update(self, target: np.ndarray | None) -> np.ndarray:
        c = self.cfg
        if target is None:                        # 跟丟：先停住，超過 hold 幀再慢慢拉回全景
            self.lost += 1
            if self.lost < c.hold_frames:
                target = self.state.copy()
            else:
                target = np.array([self.W / 2, self.H / 2, self.W], dtype=np.float64)
        else:
            self.lost = 0
            target = np.asarray(target, dtype=np.float64)

        err = target - self.state
        w = self.state[2]
        dz = np.array([c.dead_xy * w, c.dead_xy * w / c.aspect, c.dead_zoom * w])
        err = np.sign(err) * np.maximum(np.abs(err) - dz, 0)

        gain = np.full(3, c.smooth)
        if err[2] > 0:                            # 需要拉遠（裁切變寬）→ 反應更快
            gain[2] *= c.zoom_out_boost
        desired = gain * err

        max_v = np.array([c.max_pan, c.max_pan, c.max_zoom]) * self.W
        max_a = max_v * c.accel_ratio
        self.vel += np.clip(desired - self.vel, -max_a, max_a)
        self.vel = np.clip(self.vel, -max_v, max_v)
        prev = self.state.copy()
        self.state += self.vel
        self._clamp()
        self.vel = self.state - prev              # 撞到畫面邊界時速度歸零，不會「卡」著慣性
        return self.crop_rect()

    def _clamp(self) -> None:
        c = self.cfg
        cw = np.clip(self.state[2], self.W / c.zoom_max, min(self.W, self.H * c.aspect))
        ch = cw / c.aspect
        self.state[2] = cw
        self.state[0] = np.clip(self.state[0], cw / 2, self.W - cw / 2)
        self.state[1] = np.clip(self.state[1], ch / 2, self.H - ch / 2)

    def crop_rect(self) -> np.ndarray:
        cx, cy, cw = self.state
        ch = cw / self.cfg.aspect
        return np.array([cx - cw / 2, cy - ch / 2, cx + cw / 2, cy + ch / 2])

    @property
    def zoom(self) -> float:
        return self.W / self.state[2]

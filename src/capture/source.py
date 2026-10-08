"""影像來源：影片檔（同步、不丟幀）或攝影機（獨立執行緒、永遠取最新幀）。"""
from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np


def is_camera_index(src: str | int) -> bool:
    return isinstance(src, int) or str(src).isdigit()


class VideoSource:
    """`for idx, frame in VideoSource(src): ...`

    - 影片檔：同步讀取，不可丟幀（否則輸出時間軸會和原音軌對不上）。
    - 攝影機：背景執行緒讀取、佇列長度 1，新幀覆蓋舊幀，避免延遲累積。

    每次產出一幀後，`last_timestamp`（秒）與 `last_capture_t`（perf_counter）會更新，
    供計算時間軸與端到端延遲。
    """

    def __init__(self, src: str | int, camera_cfg=None):
        self.is_live = is_camera_index(src)
        self.path: str | None = None
        self.frame_count: int | None = None
        self.last_timestamp = 0.0
        self.last_capture_t = 0.0
        self._reader: _LatestFrameReader | None = None

        if self.is_live:
            self.cap = self._open_camera(int(src), camera_cfg)
        else:
            path = Path(src)
            if not path.is_file():
                raise FileNotFoundError(f"input video not found: {path}")
            self.path = str(path)
            self.cap = cv2.VideoCapture(self.path)
            if not self.cap.isOpened():
                raise RuntimeError(f"cannot open video: {path}")
            n = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.frame_count = n if n > 0 else None

        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if self.is_live:
            # 攝影機 / 擷取卡回報的 FPS 常常是 0 或亂填，以設定檔為準
            cfg_fps = float(getattr(camera_cfg, "fps", 30.0) or 30.0)
            if not (1.0 <= fps <= 240.0) or abs(fps - cfg_fps) > 1.0:
                fps = cfg_fps
        elif not (1.0 <= fps <= 240.0):
            print(f"[capture] video reports fps={fps}, assuming 30", file=sys.stderr)
            fps = 30.0
        self.fps = fps

        if self.is_live:
            self._reader = _LatestFrameReader(self.cap)
            self._reader.start()
            self._t0 = time.perf_counter()

    @staticmethod
    def _open_camera(index: int, cfg) -> cv2.VideoCapture:
        # Linux 擷取卡通常是 V4L2；Windows 用 DirectShow 開啟較快且支援 MJPG
        api = {"linux": cv2.CAP_V4L2, "win32": cv2.CAP_DSHOW}.get(sys.platform, cv2.CAP_ANY)
        cap = cv2.VideoCapture(index, api)
        if not cap.isOpened():
            cap = cv2.VideoCapture(index)
        if not cap.isOpened():
            raise RuntimeError(f"cannot open camera {index}")
        if cfg is not None:
            # 明確指定 MJPG + 解析度，否則可能預設成低解析度 YUYV
            if cfg.fourcc:
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*cfg.fourcc))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.height)
            cap.set(cv2.CAP_PROP_FPS, cfg.fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

    def __iter__(self):
        idx = 0
        if self.is_live:
            seq = 0
            while True:
                got = self._reader.get(seq)
                if got is None:
                    return
                seq, frame, t = got
                self.last_capture_t = t
                self.last_timestamp = t - self._t0
                yield idx, frame
                idx += 1
        else:
            while True:
                ok, frame = self.cap.read()
                if not ok:
                    return
                self.last_capture_t = time.perf_counter()
                self.last_timestamp = idx / self.fps
                yield idx, frame
                idx += 1

    def release(self) -> None:
        if self._reader is not None:
            self._reader.stop()
        self.cap.release()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.release()

    def __repr__(self) -> str:
        kind = "camera" if self.is_live else self.path
        return f"VideoSource({kind}, {self.width}x{self.height} @ {self.fps:.2f} fps)"


class _LatestFrameReader(threading.Thread):
    """持續讀取攝影機，只保留最新一幀。"""

    def __init__(self, cap: cv2.VideoCapture, max_failures: int = 100):
        super().__init__(daemon=True)
        self.cap = cap
        self.max_failures = max_failures
        self.cond = threading.Condition()
        self.frame: np.ndarray | None = None
        self.t = 0.0
        self.seq = 0
        self.stopped = False

    def run(self) -> None:
        failures = 0
        while not self.stopped:
            ok, frame = self.cap.read()
            t = time.perf_counter()
            if not ok:
                failures += 1
                if failures > self.max_failures:
                    break
                time.sleep(0.01)
                continue
            failures = 0
            with self.cond:
                self.frame, self.t = frame, t
                self.seq += 1
                self.cond.notify_all()
        with self.cond:
            self.stopped = True
            self.cond.notify_all()

    def get(self, last_seq: int, timeout: float = 5.0):
        """等待比 last_seq 更新的一幀；攝影機中斷時回傳 None。"""
        with self.cond:
            self.cond.wait_for(lambda: self.seq != last_seq or self.stopped, timeout)
            if self.seq == last_seq:
                return None
            return self.seq, self.frame, self.t

    def stop(self) -> None:
        with self.cond:
            self.stopped = True
            self.cond.notify_all()
        self.join(timeout=1.0)

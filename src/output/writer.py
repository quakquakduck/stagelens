"""影片輸出：ffmpeg pipe（H.264 + 原音軌，一次完成），找不到 ffmpeg 時退回 OpenCV（無聲）。

OpenCV 的 VideoWriter 不會保留音訊；舞台演出沒有音樂就失去意義，所以優先使用 ffmpeg：
原始幀以 rawvideo 經 stdin 送入，同時以第二個輸入讀原檔音軌。
Stage II 的 RTMP 直播也是同一套做法（輸出改成 `-f flv rtmp://...`）。
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from fractions import Fraction
from pathlib import Path

import cv2
import numpy as np


def find_ffmpeg(explicit: str | None = None) -> str | None:
    """指定路徑 → PATH 上的 ffmpeg → imageio-ffmpeg 內附的執行檔。"""
    if explicit:
        return explicit
    found = shutil.which("ffmpeg")
    if found:
        return found
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def fps_fraction(fps: float) -> str:
    """29.97 → '30000/1001'，避免長片累積時間誤差。"""
    f = Fraction(fps).limit_denominator(1001)
    return f"{f.numerator}/{f.denominator}"


class FFmpegWriter:
    def __init__(self, path: str, size: tuple[int, int], fps: float, audio_src: str | None = None,
                 crf: int = 18, preset: str = "veryfast", ffmpeg: str = "ffmpeg"):
        w, h = size
        self.path, self.size = path, (w, h)
        cmd = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
               "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", fps_fraction(fps), "-i", "-"]
        if audio_src:
            # 1:a:0? → 原檔沒有音軌時也不會失敗；-shortest 防止長度不一
            cmd += ["-i", audio_src, "-map", "0:v:0", "-map", "1:a:0?", "-c:a", "aac", "-b:a", "192k",
                    "-shortest"]
        cmd += ["-c:v", "libx264", "-preset", preset, "-crf", str(crf), "-pix_fmt", "yuv420p"]
        if path.lower().endswith((".mp4", ".mov", ".m4v")):
            cmd += ["-movflags", "+faststart"]
        cmd.append(path)
        self._log = tempfile.TemporaryFile()
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=self._log)
        self.frames = 0

    def write(self, frame: np.ndarray, repeat: int = 1) -> None:
        if frame.shape[1::-1] != self.size:
            raise ValueError(f"frame size {frame.shape[1::-1]} != writer size {self.size}")
        data = np.ascontiguousarray(frame).data
        try:
            for _ in range(repeat):
                self.proc.stdin.write(data)
        except (BrokenPipeError, OSError):
            self.proc.wait()
            raise RuntimeError(f"ffmpeg exited unexpectedly:\n{self._stderr()}") from None
        self.frames += repeat

    def close(self) -> None:
        if self.proc.stdin and not self.proc.stdin.closed:
            try:
                self.proc.stdin.close()
            except OSError:
                pass
        code = self.proc.wait()
        err = self._stderr()
        self._log.close()
        if code != 0:
            raise RuntimeError(f"ffmpeg failed (exit {code}):\n{err}")

    def _stderr(self) -> str:
        self._log.seek(0)
        return self._log.read().decode(errors="replace")[-2000:]


class CvWriter:
    """備援：OpenCV VideoWriter（mp4v，沒有聲音）。之後可用 scripts/mux_audio.sh 接回音軌。"""

    def __init__(self, path: str, size: tuple[int, int], fps: float):
        self.path, self.size = path, size
        self.vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
        if not self.vw.isOpened():
            raise RuntimeError(f"cannot open VideoWriter for {path}")
        self.frames = 0

    def write(self, frame: np.ndarray, repeat: int = 1) -> None:
        for _ in range(repeat):
            self.vw.write(frame)
        self.frames += repeat

    def close(self) -> None:
        self.vw.release()


def make_writer(path: str | None, size: tuple[int, int], fps: float, audio_src: str | None = None,
                encoder_cfg=None):
    """建立輸出；path 為 None 時回傳 None。輸出 FPS 必須等於輸入 FPS，否則音畫不同步。"""
    if not path:
        return None
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    ffmpeg = find_ffmpeg(getattr(encoder_cfg, "ffmpeg", None))
    if ffmpeg:
        return FFmpegWriter(path, size, fps, audio_src,
                            crf=getattr(encoder_cfg, "crf", 18),
                            preset=getattr(encoder_cfg, "preset", "veryfast"),
                            ffmpeg=ffmpeg)
    print("[output] ffmpeg not found (install ffmpeg or `pip install imageio-ffmpeg`); "
          "writing with OpenCV — the output will have NO AUDIO", file=sys.stderr)
    return CvWriter(path, size, fps)

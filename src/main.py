"""StageLens CLI 進入點。

    python src/main.py --input samples/demo.mp4 --view split --debug
    python src/main.py --input 0 --output out/stage.mp4
    python src/main.py --input samples/demo.mp4 --no-lighting
"""
import os
import sys

if __package__ in (None, ""):
    # 以 `python src/main.py` 執行時：讓 repo 根目錄可被 import（`import src.*`），
    # 並把 src/ 本身移出 sys.path，避免 src/ 底下的模組名稱遮蔽同名的套件。
    _SRC_DIR = os.path.dirname(os.path.abspath(__file__))
    sys.path[:] = [p for p in sys.path if os.path.abspath(p or os.curdir) != _SRC_DIR]
    sys.path.insert(0, os.path.dirname(_SRC_DIR))

import time

import cv2
from tqdm import tqdm

from src.capture import VideoSource
from src.config import Config, load_config
from src.output import make_writer
from src.pipeline import Pipeline

WINDOW = "StageLens"


def run(cfg: Config) -> int:
    if cfg.input is None:
        print("error: --input is required (video file path or camera index)", file=sys.stderr)
        return 2

    src = VideoSource(cfg.input, cfg.camera)
    print(f"[main] {src}", file=sys.stderr)
    pipe = Pipeline(cfg, src.width, src.height, src.fps)
    # 影片檔模式把原檔音軌接回去；攝影機模式的音訊（擷取卡音訊裝置）留待 Stage II
    writer = make_writer(cfg.output, cfg.output_size, src.fps,
                         audio_src=None if src.is_live else src.path, encoder_cfg=cfg.encoder)
    show = cfg.show and _display_available()

    total = src.frame_count
    if cfg.max_frames:
        total = min(total, cfg.max_frames) if total else cfg.max_frames
    bar = tqdm(total=total, unit="frame", dynamic_ncols=True, disable=src.is_live and show)

    n = written = 0
    t_start = time.perf_counter()
    try:
        for idx, frame in src:
            if cfg.max_frames and idx >= cfg.max_frames:
                break
            view, _ = pipe.process(idx, frame, src.last_timestamp)

            if writer is not None:
                with pipe.timer("write"):
                    if src.is_live:
                        # 攝影機模式會丟幀：依實際經過時間補寫重複幀，讓輸出長度等於真實時間
                        due = int((time.perf_counter() - t_start) * src.fps) + 1
                        if due > written:
                            writer.write(view, repeat=due - written)
                            written = due
                    else:
                        writer.write(view)
            pipe.timer.end_frame(frame=idx, latency=(time.perf_counter() - src.last_capture_t) * 1000.0)
            n += 1
            bar.update(1)
            if n % 15 == 0:
                bar.set_postfix(fps=f"{pipe.timer.fps():.1f}", zoom=f"{pipe.camera.zoom:.2f}")

            if show:
                show = _show(view, paused_ok=not src.is_live)
                if show is None:          # 使用者按 q / Esc
                    break
    except KeyboardInterrupt:
        print("\n[main] interrupted, finalizing output…", file=sys.stderr)
    finally:
        bar.close()
        src.release()
        pipe.close()
        if writer is not None:
            writer.close()
        if cfg.show:
            cv2.destroyAllWindows()

    elapsed = time.perf_counter() - t_start
    if cfg.timings_csv and pipe.timer.rows:
        pipe.timer.dump_csv(cfg.timings_csv)
    _print_summary(pipe, n, elapsed, cfg)
    return 0


def _display_available() -> bool:
    if sys.platform.startswith("linux") and not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        print("[main] no display found, preview window disabled", file=sys.stderr)
        return False
    return True


_window_ready = False


def _show(view, paused_ok: bool):
    """顯示一幀。回傳 True = 繼續顯示、False = 視窗無法使用、None = 使用者要求結束。"""
    global _window_ready
    try:
        if not _window_ready:
            cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
            h, w = view.shape[:2]
            s = min(1.0, 1280 / w)
            cv2.resizeWindow(WINDOW, int(w * s), int(h * s))
            _window_ready = True
        cv2.imshow(WINDOW, view)
        key = cv2.waitKey(1) & 0xFF
        if key == ord(" ") and paused_ok:     # 空白鍵暫停 / 繼續
            while (key := cv2.waitKey(0) & 0xFF) not in (ord(" "), ord("q"), 27):
                pass
        if key in (ord("q"), 27):
            return None
        return True
    except cv2.error as e:
        print(f"[main] preview window unavailable ({e.err}); continuing without it", file=sys.stderr)
        return False


def _print_summary(pipe: Pipeline, n: int, elapsed: float, cfg: Config) -> None:
    if n == 0:
        print("[main] no frames processed", file=sys.stderr)
        return
    print(f"\nprocessed {n} frames in {elapsed:.1f} s ({n / elapsed:.2f} FPS)", file=sys.stderr)
    summary = pipe.timer.summary(skip=min(10, n // 10))
    print(f"  {'stage':<10}{'median':>9}{'p95':>9}  (ms)", file=sys.stderr)
    for name, s in summary.items():
        if name != "frame":
            print(f"  {name:<10}{s['median']:>9.1f}{s['p95']:>9.1f}", file=sys.stderr)
    if cfg.timings_csv:
        print(f"timings → {cfg.timings_csv}", file=sys.stderr)
    if cfg.output:
        print(f"output  → {cfg.output}", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    try:
        return run(load_config(argv))
    except (FileNotFoundError, NotImplementedError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())

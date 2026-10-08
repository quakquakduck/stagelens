"""固定影片的效能 / 品質量測：各階段耗時 CSV + 摘要 JSON + 可貼進 docs/benchmarks.md 的 Markdown。

    python scripts/benchmark.py --input samples/demo.mp4
    python scripts/benchmark.py --input samples/demo.mp4 --backend cuda --max-frames 600
    python scripts/benchmark.py --input samples/demo.mp4 --pipelines full --save-video

預設依序量測 README Performance 表的兩種 pipeline：
  framing = 偵測 + 自動運鏡（燈光還原關閉）
  full    = 偵測 + 自動運鏡 + 燈光還原
所有 main.py 的參數（--detect-every、--mode、--set ...）都可以使用。
Stage II 用同一支影片、同樣參數跑 --backend hailo，數字即可直接對比。
"""
from __future__ import annotations

import copy
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from tqdm import tqdm  # noqa: E402

from src.capture import VideoSource  # noqa: E402
from src.config import build_parser, config_from_args  # noqa: E402
from src.detection import make_backend  # noqa: E402
from src.output import make_writer  # noqa: E402
from src.pipeline import Pipeline  # noqa: E402
from src.utils.metrics import QualityMetrics  # noqa: E402

PIPELINES = {"framing": "framing only", "full": "framing + lighting"}


def main() -> int:
    ap = build_parser("StageLens benchmark")
    ap.add_argument("--pipelines", default="framing,full", help="comma-separated: framing, full")
    ap.add_argument("--warmup", type=int, default=10, help="frames excluded from statistics")
    ap.add_argument("--out-dir", default="out/bench")
    ap.add_argument("--tag", help="file name prefix (default: timestamp)")
    ap.add_argument("--save-video", action="store_true", help="also encode the output video (adds encoder cost)")
    args = ap.parse_args()
    cfg = config_from_args(args)
    cfg.show = False
    if cfg.input is None or cfg.input.isdigit():
        ap.error("--input must be a video file (benchmarks need a fixed, repeatable input)")
    names = [p.strip() for p in args.pipelines.split(",") if p.strip()]
    unknown = [p for p in names if p not in PIPELINES]
    if unknown:
        ap.error(f"unknown pipeline(s): {unknown}; choose from {list(PIPELINES)}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = args.tag or datetime.now().strftime("%Y%m%d-%H%M%S")

    backend = make_backend(cfg)            # 各 pipeline 共用同一個模型，避免重複載入
    system = system_info()
    results = {"tag": tag, "system": system, "config": cfg.to_dict(), "runs": {}}
    try:
        for name in names:
            c = copy.deepcopy(cfg)
            c.lighting.enabled = name == "full"
            video = str(out_dir / f"{tag}_{name}.mp4") if args.save_video else None
            results["runs"][name] = run_one(c, backend, args.warmup, out_dir / f"{tag}_{name}_timings.csv", video)
    finally:
        backend.close()

    json_path = out_dir / f"{tag}_summary.json"
    json_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    md = to_markdown(results, cfg)
    md_path = out_dir / f"{tag}_summary.md"
    md_path.write_text(md, encoding="utf-8")
    print("\n" + md)
    print(f"summary → {json_path}\nmarkdown → {md_path}", file=sys.stderr)
    return 0


def run_one(cfg, backend, warmup: int, csv_path: Path, video_path: str | None) -> dict:
    src = VideoSource(cfg.input)
    pipe = Pipeline(cfg, src.width, src.height, src.fps, backend=backend, collect_stats=True)
    metrics = QualityMetrics(src.width, src.height, cfg.framing)
    writer = make_writer(video_path, cfg.output_size, src.fps, src.path, cfg.encoder) if video_path else None
    total = min(src.frame_count or 0, cfg.max_frames or 10**9) or None
    label = PIPELINES["full" if cfg.lighting.enabled else "framing"]

    n, t_measure = 0, None
    t_prev = time.perf_counter()
    try:
        for idx, frame in tqdm(src, total=total, desc=label, unit="frame"):
            if cfg.max_frames and idx >= cfg.max_frames:
                break
            if idx == warmup:
                t_measure = time.perf_counter()
            read_ms = (time.perf_counter() - t_prev) * 1000.0
            view, res = pipe.process(idx, frame, src.last_timestamp)
            if writer is not None:
                with pipe.timer("write"):
                    writer.write(view)
            res.timings = pipe.timer.end_frame(
                frame=idx, read=read_ms, latency=(time.perf_counter() - src.last_capture_t) * 1000.0)
            metrics.update(res)
            n += 1
            t_prev = time.perf_counter()
    finally:
        src.release()
        if writer is not None:
            writer.close()
    elapsed = time.perf_counter() - t_measure if t_measure else None

    pipe.timer.dump_csv(csv_path)
    measured = n - warmup if t_measure else 0
    return {
        "pipeline": label,
        "input": f"{src.width}x{src.height}",
        "input_fps": src.fps,
        "frames": n,
        "measured_frames": measured,
        "fps": measured / elapsed if elapsed else None,
        "stages_ms": pipe.timer.summary(skip=min(warmup, n)),
        "quality": metrics.summary(),
        "timings_csv": str(csv_path),
        "video": video_path,
    }


def system_info() -> dict:
    import onnxruntime as ort

    info = {
        "cpu": cpu_model(),
        "cores": os.cpu_count(),
        "os": platform.platform(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "opencv": cv2.__version__,
        "onnxruntime": ort.__version__,
        "providers": ort.get_available_providers(),
    }
    try:
        gpu = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=5)
        if gpu.returncode == 0 and gpu.stdout.strip():
            info["gpu"] = gpu.stdout.strip().splitlines()
    except (OSError, subprocess.TimeoutExpired):
        pass
    return info


def cpu_model() -> str:
    try:
        if sys.platform.startswith("linux"):
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
        elif sys.platform == "darwin":
            return subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                                  capture_output=True, text=True).stdout.strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def to_markdown(results: dict, cfg) -> str:
    sysinfo = results["system"]
    platform_name = f"{sysinfo['cpu']} ({cfg.backend})"
    lines = [
        f"### Benchmark {results['tag']}",
        "",
        f"- Platform: {sysinfo['cpu']}, {sysinfo['cores']} threads, {sysinfo['os']}"
        + (f", GPU: {', '.join(sysinfo['gpu'])}" if sysinfo.get("gpu") else ""),
        f"- Software: Python {sysinfo['python']}, OpenCV {sysinfo['opencv']}, onnxruntime {sysinfo['onnxruntime']}",
        f"- Input: `{cfg.input}`, output {cfg.output_size[0]}x{cfg.output_size[1]}, view `{cfg.view}`",
        f"- Model: {cfg.detection.model} / {cfg.detection.mode}, backend `{cfg.backend}`, "
        f"detect_every={cfg.detect_every}, infer_long_side={cfg.infer_long_side}",
        "",
        "| Platform | Input | Pipeline | FPS | End-to-end latency (median / P95) |",
        "|---|---|---|---|---|",
    ]
    for run in results["runs"].values():
        lat = run["stages_ms"].get("latency", {})
        fps = f"{run['fps']:.1f}" if run["fps"] else "n/a"
        lat_s = f"{lat['median']:.0f} / {lat['p95']:.0f} ms" if lat else "n/a"
        lines.append(f"| {platform_name} | {run['input']} | {run['pipeline']} | {fps} | {lat_s} |")

    lines += ["", "| Pipeline | Stage | median ms | P95 ms | samples |", "|---|---|---|---|---|"]
    for run in results["runs"].values():
        for stage, s in run["stages_ms"].items():
            if stage not in ("frame", "latency"):
                lines.append(f"| {run['pipeline']} | {stage} | {s['median']:.1f} | {s['p95']:.1f} | {s['n']} |")

    def pct(v):
        return f"{v * 100:.1f}%" if v is not None else "n/a"

    def num(v, unit=""):
        return f"{v:.2f}{unit}" if v is not None else "n/a"

    lines += ["", "| Pipeline | Subject completeness | Smoothness (px/frame², lower is better) | Mean zoom "
              "| Face over-exposed before → after |", "|---|---|---|---|---|"]
    for run in results["runs"].values():
        q = run["quality"]
        face = (f"{pct(q['face_overexposed_before'])} → {pct(q['face_overexposed_after'])}"
                if q["face_overexposed_after"] is not None and run["pipeline"] == PIPELINES["full"] else "—")
        lines.append(f"| {run['pipeline']} | {pct(q['subject_complete_ratio'])} | {num(q['smoothness_px'])} "
                     f"| {num(q['mean_zoom'], 'x')} | {face} |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    sys.exit(main())

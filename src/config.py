"""設定：dataclass 預設值 ← configs/default.yaml ← --config ← CLI 參數 ← --set（後者覆寫前者）。"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "default.yaml"

VIEWS = ("output", "split", "debug")
BACKENDS = ("cpu", "cuda", "hailo")
POSE_MODELS = ("rtmpose", "rtmo")
POSE_MODES = ("lightweight", "balanced", "performance")
INTERPS = ("linear", "cubic")


@dataclass
class DetectionConfig:
    model: str = "rtmpose"
    mode: str = "balanced"
    models_dir: str = "models"


@dataclass
class TrackerConfig:
    iou_thr: float = 0.3
    max_age: int = 15


@dataclass
class CameraConfig:
    width: int = 1920
    height: int = 1080
    fps: float = 30.0
    fourcc: str = "MJPG"


@dataclass
class FramingConfig:
    aspect: float | None = None
    zoom_max: float = 2.5
    smooth: float = 0.15
    zoom_out_boost: float = 3.0
    dead_xy: float = 0.05
    dead_zoom: float = 0.08
    max_pan: float = 0.02
    max_zoom: float = 0.015
    accel_ratio: float = 0.15
    hold_frames: int = 30
    burst_window_s: float = 0.7
    min_person_h: float = 0.08
    spread_full: float = 0.8
    pad_top: float = 0.15
    pad_bottom: float = 0.08
    pad_side: float = 0.15
    interp: str = "cubic"


@dataclass
class LightingConfig:
    enabled: bool = True
    highlight_thr: float = 0.92
    knee: float = 0.70
    compress: float = 0.5
    clahe_clip: float = 1.5
    clahe_tiles: int = 4
    detail: float = 0.5
    strength_ema: float = 0.1
    feather: float = 0.15
    min_face_px: int = 8


@dataclass
class EncoderConfig:
    crf: int = 18
    preset: str = "veryfast"
    ffmpeg: str | None = None


@dataclass
class Config:
    input: str | None = None
    output: str | None = None
    view: str = "output"
    debug: bool = False
    show: bool = True
    backend: str = "cpu"
    output_size: tuple[int, int] = (1920, 1080)
    detect_every: int = 2
    infer_long_side: int = 960
    max_frames: int | None = None
    timings_csv: str | None = "out/timings.csv"
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    framing: FramingConfig = field(default_factory=FramingConfig)
    lighting: LightingConfig = field(default_factory=LightingConfig)
    encoder: EncoderConfig = field(default_factory=EncoderConfig)

    @property
    def no_lighting(self) -> bool:
        return not self.lighting.enabled

    @property
    def models_dir(self) -> Path:
        p = Path(self.detection.models_dir)
        return p if p.is_absolute() else REPO_ROOT / p

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# CLI 參數名稱 → 設定欄位（點號表示巢狀）
_CLI_KEYS = {
    "input": "input",
    "output": "output",
    "view": "view",
    "debug": "debug",
    "show": "show",
    "backend": "backend",
    "zoom_max": "framing.zoom_max",
    "smooth": "framing.smooth",
    "detect_every": "detect_every",
    "infer_long_side": "infer_long_side",
    "model": "detection.model",
    "mode": "detection.mode",
    "max_frames": "max_frames",
    "output_size": "output_size",
}


def build_parser(description: str = "StageLens — AI edge director for stage performances") -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--config", help="extra YAML config merged over configs/default.yaml")
    p.add_argument("--input", "-i", help="video file path or camera index")
    p.add_argument("--output", "-o", help="output video path (default: none)")
    p.add_argument("--view", choices=VIEWS, help="output | split (raw vs. directed) | debug")
    p.add_argument("--debug", action="store_true", default=None,
                   help="draw skeletons, boxes, crop window, FPS/latency")
    p.add_argument("--no-lighting", action="store_true", help="disable lighting restoration")
    p.add_argument("--backend", choices=BACKENDS, help="inference backend (hailo = Stage II)")
    p.add_argument("--zoom-max", type=float, help="maximum virtual zoom factor (default 2.5)")
    p.add_argument("--smooth", type=float, help="camera path smoothing, lower = smoother (default 0.15)")
    p.add_argument("--detect-every", type=int, help="run pose estimation every N frames")
    p.add_argument("--infer-long-side", type=int, help="downscale long side before inference (0 = off)")
    p.add_argument("--model", choices=POSE_MODELS, help="pose model family")
    p.add_argument("--mode", choices=POSE_MODES, help="model size")
    p.add_argument("--output-size", type=_parse_size, metavar="WxH", help="output resolution, e.g. 1920x1080")
    p.add_argument("--max-frames", type=int, help="stop after N frames")
    p.add_argument("--show", action=argparse.BooleanOptionalAction, default=None,
                   help="show a preview window (default: on when a display is available)")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                   help="override any config field, e.g. --set framing.dead_xy=0.08 (repeatable)")
    return p


def config_from_args(args: argparse.Namespace) -> Config:
    data = _read_yaml(DEFAULT_CONFIG) if DEFAULT_CONFIG.exists() else {}
    if args.config:
        _deep_update(data, _read_yaml(Path(args.config)))

    for arg_name, key in _CLI_KEYS.items():
        value = getattr(args, arg_name, None)
        if value is not None:
            _set_dotted(data, key, value)
    if getattr(args, "no_lighting", False):
        _set_dotted(data, "lighting.enabled", False)
    for item in getattr(args, "set", None) or []:
        if "=" not in item:
            raise SystemExit(f"--set expects KEY=VALUE, got {item!r}")
        key, raw = item.split("=", 1)
        _set_dotted(data, key.strip(), yaml.safe_load(raw))

    try:
        cfg = _from_dict(Config, data)
        _finalize(cfg)
    except ValueError as e:
        raise SystemExit(f"config error: {e}") from None
    return cfg


def load_config(argv: list[str] | None = None) -> Config:
    return config_from_args(build_parser().parse_args(argv))


def _parse_size(text: str) -> tuple[int, int]:
    try:
        w, h = text.lower().split("x")
        return int(w), int(h)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected WxH, got {text!r}") from None


def _read_yaml(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise SystemExit(f"{path}: top level must be a mapping")
    return data


def _deep_update(base: dict, upd: dict) -> dict:
    for k, v in upd.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v
    return base


def _set_dotted(data: dict, key: str, value: Any) -> None:
    *parents, last = key.split(".")
    node = data
    for p in parents:
        node = node.setdefault(p, {})
    node[last] = value


def _from_dict(cls, data: dict, prefix: str = ""):
    obj = cls()
    names = {f.name for f in fields(cls)}
    for k, v in data.items():
        if k not in names:
            raise ValueError(f"unknown key '{prefix}{k}'")
        current = getattr(obj, k)
        if is_dataclass(current):
            if not isinstance(v, dict):
                raise ValueError(f"'{prefix}{k}' must be a mapping")
            v = _from_dict(type(current), v, f"{prefix}{k}.")
        setattr(obj, k, v)
    return obj


def _finalize(cfg: Config) -> None:
    def check(name: str, value: Any, choices: tuple) -> None:
        if value not in choices:
            raise ValueError(f"{name} must be one of {choices}, got {value!r}")

    check("view", cfg.view, VIEWS)
    check("backend", cfg.backend, BACKENDS)
    check("detection.model", cfg.detection.model, POSE_MODELS)
    check("detection.mode", cfg.detection.mode, POSE_MODES)
    check("framing.interp", cfg.framing.interp, INTERPS)

    if cfg.input is not None:
        cfg.input = str(cfg.input)
    w, h = (int(v) for v in cfg.output_size)
    if w <= 0 or h <= 0 or w % 2 or h % 2:
        raise ValueError(f"output_size must be positive even numbers (H.264 / yuv420p), got {w}x{h}")
    cfg.output_size = (w, h)
    if cfg.framing.aspect is None:
        cfg.framing.aspect = w / h
    elif abs(cfg.framing.aspect - w / h) > 0.01:
        # 裁切框比例與輸出比例不同會讓畫面被拉伸
        raise ValueError(f"framing.aspect ({cfg.framing.aspect}) does not match output_size {w}x{h}")
    if int(cfg.detect_every) < 1:
        raise ValueError("detect_every must be >= 1")
    cfg.detect_every = int(cfg.detect_every)
    if cfg.framing.zoom_max < 1:
        raise ValueError("framing.zoom_max must be >= 1")

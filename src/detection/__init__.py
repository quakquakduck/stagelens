from .base import PoseBackend, ScaledBackend, persons_from_arrays
from .tracker import Tracker


def make_backend(cfg) -> PoseBackend:
    """依 cfg.backend 建立姿態後端；cfg.infer_long_side > 0 時外包一層降解析度。"""
    if cfg.backend in ("cpu", "cuda"):
        from .rtmlib_backend import RtmlibBackend

        backend = RtmlibBackend(device=cfg.backend, mode=cfg.detection.mode,
                                model=cfg.detection.model, models_dir=cfg.models_dir)
    elif cfg.backend == "hailo":
        from .hailo_backend import HailoBackend

        backend = HailoBackend()
    else:
        raise ValueError(f"unknown backend: {cfg.backend}")
    if cfg.infer_long_side and cfg.infer_long_side > 0:
        backend = ScaledBackend(backend, cfg.infer_long_side)
    return backend


__all__ = ["PoseBackend", "ScaledBackend", "Tracker", "make_backend", "persons_from_arrays"]

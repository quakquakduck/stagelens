"""cpu / cuda 後端：透過 rtmlib 跑 ONNX（YOLOX + RTMPose 兩階段，或 RTMO 單階段）。"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from ..datatypes import Person
from .base import PoseBackend, persons_from_arrays


def model_specs(model: str, mode: str) -> dict[str, tuple[str, tuple[int, int]]]:
    """回傳 {角色: (ONNX 下載網址, 輸入尺寸)}；取自 rtmlib 內建表，與已安裝版本同步。"""
    from rtmlib import Body

    if model == "rtmo":
        m = Body.RTMO_MODE[mode]
        return {"pose": (m["pose"], tuple(m["pose_input_size"]))}
    m = Body.MODE[mode]
    return {
        "det": (m["det"], tuple(m["det_input_size"])),
        "pose": (m["pose"], tuple(m["pose_input_size"])),
    }


def local_model_path(url: str, models_dir: str | Path) -> Path:
    """scripts/download_models.py 解壓後的檔名：<zip 檔名去掉副檔名>.onnx。"""
    name = url.rsplit("/", 1)[-1].split(".")[0]
    return Path(models_dir) / f"{name}.onnx"


class RtmlibBackend(PoseBackend):
    def __init__(self, device: str = "cpu", mode: str = "balanced", model: str = "rtmpose",
                 models_dir: str | Path = "models"):
        from rtmlib import RTMO, YOLOX, RTMPose

        device = _check_device(device)
        specs = model_specs(model, mode)

        def resolve(url: str) -> str:
            path = local_model_path(url, models_dir)
            if path.exists():
                return str(path)
            print(f"[detection] {path} not found, rtmlib will download it to its cache "
                  "(run scripts/download_models.py to pre-fetch)", file=sys.stderr)
            return url

        common = dict(backend="onnxruntime", device=device)
        if model == "rtmo":
            url, size = specs["pose"]
            self.det = None
            self.pose = RTMO(resolve(url), model_input_size=size, **common)
        else:
            det_url, det_size = specs["det"]
            pose_url, pose_size = specs["pose"]
            self.det = YOLOX(resolve(det_url), model_input_size=det_size, **common)
            self.pose = RTMPose(resolve(pose_url), model_input_size=pose_size, **common)
        self.name = f"{model}-{mode}@{device}"

    def infer(self, frame_bgr: np.ndarray) -> list[Person]:
        if self.det is None:
            kpts, scores = self.pose(frame_bgr)
        else:
            bboxes = self.det(frame_bgr)
            if len(bboxes) == 0:   # 沒有框時 RTMPose 會把整張畫面當成一個人，必須跳過
                return []
            kpts, scores = self.pose(frame_bgr, bboxes=bboxes)
        return persons_from_arrays(kpts, scores)


def _check_device(device: str) -> str:
    if device == "cuda":
        import onnxruntime as ort

        if "CUDAExecutionProvider" not in ort.get_available_providers():
            print("[detection] CUDAExecutionProvider is not available "
                  "(pip uninstall onnxruntime && pip install onnxruntime-gpu); falling back to CPU",
                  file=sys.stderr)
            return "cpu"
    return device

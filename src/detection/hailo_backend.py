"""Stage II：ASUS UGen300（Hailo-10H）後端。

預計實作方式（詳見 IMPLEMENTATION_GUIDE.md 第 9 節）：
1. 以 Hailo Dataflow Compiler 將 Stage I 使用的同一份 ONNX 編譯成 .hef
   （parse → optimize（用真實舞台畫面校正）→ compile）。
2. 用 HailoRT（`hailo_platform`）的 VDevice + scheduler 載入偵測 / 姿態模型。
3. NPU 只跑主幹網路；YOLOX 的 NMS、RTMPose 的 SimCC 解碼、RTMO 的後處理在 CPU 以 numpy 實作。
4. 輸出 (N,17,2) / (N,17) 陣列後呼叫 `persons_from_arrays`，與 rtmlib 後端格式完全一致，
   framing / lighting / overlay / output 不需任何修改。
"""
from __future__ import annotations

import numpy as np

from ..datatypes import Person
from .base import PoseBackend


class HailoBackend(PoseBackend):
    def __init__(self, hef_dir: str | None = None, **kwargs):
        raise NotImplementedError(
            "The Hailo (UGen300) backend is planned for Stage II. "
            "Use --backend cpu or --backend cuda for now; see hardware/ugen300_port.md "
            "and IMPLEMENTATION_GUIDE.md section 9 for the porting plan.")

    def infer(self, frame_bgr: np.ndarray) -> list[Person]:
        raise NotImplementedError

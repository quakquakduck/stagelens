"""裁切 + 放大到輸出尺寸。

用帶小數位移的仿射矩陣（warpAffine）一次完成；若先 int() 取整再 slice，
慢速平移時會出現 1px 的跳動。
"""
from __future__ import annotations

import cv2
import numpy as np

INTERP = {"linear": cv2.INTER_LINEAR, "cubic": cv2.INTER_CUBIC}


def crop_region(frame: np.ndarray, crop: np.ndarray, pad: int = 2) -> tuple[np.ndarray, tuple[int, int]]:
    """涵蓋裁切框的整數像素區域（view，不複製）與其左上角座標。

    燈光修正在這個原解析度區域上做，處理的像素量最少；之後再由 warp_to_output 放大。
    """
    H, W = frame.shape[:2]
    x0, y0, x1, y1 = crop
    ix0 = max(0, int(np.floor(x0)) - pad)
    iy0 = max(0, int(np.floor(y0)) - pad)
    ix1 = min(W, int(np.ceil(x1)) + pad)
    iy1 = min(H, int(np.ceil(y1)) + pad)
    return frame[iy0:iy1, ix0:ix1], (ix0, iy0)


def warp_to_output(region: np.ndarray, crop: np.ndarray, offset: tuple[int, int],
                   out_size: tuple[int, int], interp: str = "cubic") -> np.ndarray:
    """把 region 中的裁切框（原畫面座標，offset 為 region 左上角）映射到 out_size。"""
    out_w, out_h = out_size
    ox, oy = offset
    x0, y0, x1, y1 = (float(v) for v in crop)
    x0, x1, y0, y1 = x0 - ox, x1 - ox, y0 - oy, y1 - oy
    sx, sy = (x1 - x0) / out_w, (y1 - y0) / out_h       # 每個輸出像素對應幾個來源像素

    if min(sx, sy) > 1.5:
        # 大幅縮小時 warpAffine 會產生鋸齒：先用 INTER_AREA 縮到接近輸出尺寸
        rh, rw = region.shape[:2]
        f = 1.0 / min(sx, sy)
        small = cv2.resize(region, (max(1, round(rw * f)), max(1, round(rh * f))),
                           interpolation=cv2.INTER_AREA)
        fx, fy = small.shape[1] / rw, small.shape[0] / rh
        region = small
        x0, x1, y0, y1 = x0 * fx, x1 * fx, y0 * fy, y1 * fy
        sx, sy = (x1 - x0) / out_w, (y1 - y0) / out_h

    # 輸出像素 j 的中心 (j + 0.5) → 來源連續座標 x0 + (j + 0.5)·sx → 來源像素中心座標（再 −0.5）
    M = np.array([[sx, 0.0, x0 + 0.5 * sx - 0.5],
                  [0.0, sy, y0 + 0.5 * sy - 0.5]])
    return cv2.warpAffine(region, M, (out_w, out_h),
                          flags=INTERP[interp] | cv2.WARP_INVERSE_MAP,
                          borderMode=cv2.BORDER_REPLICATE)

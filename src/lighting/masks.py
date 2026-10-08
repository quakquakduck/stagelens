"""由姿態關鍵點估計臉部區域並產生羽化遮罩。

舞台遠距離下臉很小，專用臉部偵測器（如 MediaPipe short-range）常抓不到；
關鍵點已經算好，零額外成本。
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from ..datatypes import L_EAR, L_EYE, L_SHOULDER, NOSE, R_EAR, R_EYE, R_SHOULDER, Person


@dataclass
class FaceRegion:
    cx: float
    cy: float
    width: float                  # 橢圓全寬（畫面像素）
    height: float                 # 橢圓全高 = 寬 × 1.3
    sigma: float                  # 羽化 GaussianBlur sigma
    roi: tuple[int, int, int, int]  # 含羽化範圍的整數外框 x0, y0, x1, y1（畫面座標，未裁切）


def face_region(person: Person, feather: float = 0.15, min_face_px: float = 8) -> FaceRegion | None:
    k, vis = person.keypoints, person.visible()

    # 需要鼻子或兩眼才算看得到臉（背對鏡頭時不處理）
    if not (vis[NOSE] or (vis[L_EYE] and vis[R_EYE])):
        return None
    # 中心：可見頭部關鍵點（鼻、眼、耳）的平均。只用鼻子的話，轉頭 / 側臉時橢圓會偏到臉的一側
    head = [i for i in (NOSE, L_EYE, R_EYE, L_EAR, R_EAR) if vis[i]]
    cx, cy = k[head].mean(0)

    def dist(a: int, b: int) -> float:
        return float(np.linalg.norm(k[a] - k[b]))

    # 大小：取有效項目中的最大值
    sizes = []
    if vis[L_EAR] and vis[R_EAR]:
        sizes.append(dist(L_EAR, R_EAR) * 1.4)
    if vis[L_EYE] and vis[R_EYE]:
        sizes.append(dist(L_EYE, R_EYE) * 3.0)
    if vis[L_SHOULDER] and vis[R_SHOULDER]:
        sizes.append(dist(L_SHOULDER, R_SHOULDER) * 0.45)
    if vis[NOSE]:                 # 側臉時兩耳 / 兩眼距離很小，用鼻子到耳朵的距離補
        sizes += [dist(NOSE, ear) * 1.6 for ear in (L_EAR, R_EAR) if vis[ear]]
    if not sizes:                 # 最後手段：頭約佔身高 1/8
        sizes.append(float(person.bbox[3] - person.bbox[1]) * 0.12)

    width = max(sizes)
    if width < min_face_px:
        return None
    height = width * 1.3
    sigma = max(1.0, feather * width)
    m = 3.0 * sigma
    roi = (math.floor(cx - width / 2 - m), math.floor(cy - height / 2 - m),
           math.ceil(cx + width / 2 + m), math.ceil(cy + height / 2 + m))
    return FaceRegion(float(cx), float(cy), width, height, sigma, roi)


def face_mask(face: FaceRegion, image_shape: tuple[int, ...],
              offset: tuple[int, int] = (0, 0)) -> tuple[np.ndarray, tuple[int, int, int, int]] | None:
    """在 image（左上角位於畫面座標 offset）中產生羽化橢圓遮罩。

    回傳 (mask, roi)：roi 為已裁到 image 範圍內的 x0, y0, x1, y1（image 座標），
    mask 為同尺寸 float32 0–1。臉完全不在 image 內時回傳 None。
    """
    fx0, fy0, fx1, fy1 = face.roi
    canvas = np.zeros((fy1 - fy0, fx1 - fx0), np.float32)
    box = ((face.cx - fx0, face.cy - fy0), (face.width, face.height), 0.0)
    cv2.ellipse(canvas, box, 1.0, -1, cv2.LINE_AA)
    canvas = cv2.GaussianBlur(canvas, (0, 0), face.sigma, borderType=cv2.BORDER_CONSTANT)

    ox, oy = offset
    H, W = image_shape[:2]
    x0, y0 = max(0, fx0 - ox), max(0, fy0 - oy)
    x1, y1 = min(W, fx1 - ox), min(H, fy1 - oy)
    if x1 <= x0 or y1 <= y0:
        return None
    mask = canvas[y0 - (fy0 - oy):y1 - (fy0 - oy), x0 - (fx0 - ox):x1 - (fx0 - ox)]
    return np.ascontiguousarray(mask), (x0, y0, x1, y1)

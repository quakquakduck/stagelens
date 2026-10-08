"""除錯圖層（骨架、人物框 + ID、目標框、裁切框、FPS）與輸出畫面組合（output / split / debug）。"""
from __future__ import annotations

import cv2
import numpy as np

from ..datatypes import KPT_THR, Person

# COCO-17 骨架連線：(a, b, 色彩群組) 0 = 中線、1 = 左側、2 = 右側
SKELETON = [
    (0, 1, 0), (0, 2, 0), (1, 3, 1), (2, 4, 2),
    (5, 6, 0), (5, 7, 1), (7, 9, 1), (6, 8, 2), (8, 10, 2),
    (5, 11, 1), (6, 12, 2), (11, 12, 0),
    (11, 13, 1), (13, 15, 1), (12, 14, 2), (14, 16, 2),
]
LIMB_COLORS = [(255, 255, 255), (255, 160, 0), (0, 160, 255)]   # BGR
TARGET_COLOR = (0, 220, 255)     # 目標框（虛線）
CROP_COLOR = (0, 230, 0)         # 實際裁切框（實線）
FONT = cv2.FONT_HERSHEY_SIMPLEX


def track_color(track_id: int) -> tuple[int, int, int]:
    if track_id < 0:
        return 200, 200, 200
    hue = (track_id * 47) % 180
    bgr = cv2.cvtColor(np.uint8([[[hue, 200, 255]]]), cv2.COLOR_HSV2BGR)[0, 0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


def draw_persons(img: np.ndarray, persons: list[Person], scale: float = 1.0,
                 offset: tuple[float, float] = (0, 0), kpt_thr: float = KPT_THR) -> None:
    """在 img 上畫骨架與人物框；座標以 pts·scale + offset 轉換到 img。"""
    ox, oy = offset
    t = max(1, round(img.shape[0] / 400))
    for p in persons:
        pts = p.keypoints * scale + (ox, oy)
        vis = p.scores >= kpt_thr
        for a, b, g in SKELETON:
            if vis[a] and vis[b]:
                cv2.line(img, _pt(pts[a]), _pt(pts[b]), LIMB_COLORS[g], t, cv2.LINE_AA)
        for i in np.flatnonzero(vis):
            cv2.circle(img, _pt(pts[i]), t + 1, (0, 0, 255), -1, cv2.LINE_AA)
        color = track_color(p.track_id)
        x0, y0, x1, y1 = p.bbox * scale + (ox, oy, ox, oy)
        cv2.rectangle(img, _pt((x0, y0)), _pt((x1, y1)), color, t)
        if p.track_id >= 0:
            draw_label(img, f"#{p.track_id}", (x0, y0 - 4), color, 0.4 * t)


def draw_rect(img: np.ndarray, rect: np.ndarray, color, thickness: int = 2, dashed: bool = False,
              scale: float = 1.0, offset: tuple[float, float] = (0, 0)) -> None:
    ox, oy = offset
    x0, y0, x1, y1 = np.asarray(rect, dtype=np.float64) * scale + (ox, oy, ox, oy)
    if not dashed:
        cv2.rectangle(img, _pt((x0, y0)), _pt((x1, y1)), color, thickness, cv2.LINE_AA)
        return
    dash = max(6, thickness * 5)
    for (ax, ay), (bx, by) in (((x0, y0), (x1, y0)), ((x1, y0), (x1, y1)),
                               ((x1, y1), (x0, y1)), ((x0, y1), (x0, y0))):
        length = float(np.hypot(bx - ax, by - ay))
        for s in np.arange(0, length, dash * 2):
            e = min(s + dash, length)
            p0 = (ax + (bx - ax) * s / length, ay + (by - ay) * s / length)
            p1 = (ax + (bx - ax) * e / length, ay + (by - ay) * e / length)
            cv2.line(img, _pt(p0), _pt(p1), color, thickness, cv2.LINE_AA)


def draw_label(img: np.ndarray, text: str, org: tuple[float, float], color=(255, 255, 255),
               font_scale: float = 0.6, bg=(0, 0, 0)) -> None:
    """左下角位於 org 的文字，帶半透明底色。"""
    thick = max(1, round(font_scale * 2))
    (tw, th), base = cv2.getTextSize(text, FONT, font_scale, thick)
    x, y = int(org[0]), int(max(th + 2, org[1]))
    x0, y0, x1, y1 = max(0, x - 3), max(0, y - th - 4), min(img.shape[1], x + tw + 3), min(img.shape[0], y + base + 2)
    if x1 > x0 and y1 > y0:
        img[y0:y1, x0:x1] = (img[y0:y1, x0:x1] * 0.4 + np.array(bg) * 0.6).astype(np.uint8)
    cv2.putText(img, text, (x, y), FONT, font_scale, color, thick, cv2.LINE_AA)


def draw_hud(img: np.ndarray, lines: list[str], origin: tuple[int, int] | None = None) -> None:
    fs = max(0.45, img.shape[0] / 1080 * 0.8)
    step = int(32 * fs)
    x, y = origin if origin is not None else (int(12 * fs), int(12 * fs) + step)
    for i, line in enumerate(lines):
        draw_label(img, line, (x, y + i * step), (255, 255, 255), fs)


def fit_into(img: np.ndarray, w: int, h: int) -> tuple[np.ndarray, float, tuple[int, int]]:
    """等比例縮放並置中貼到 w×h 黑底（letterbox）。回傳 (canvas, scale, offset)。"""
    ih, iw = img.shape[:2]
    s = min(w / iw, h / ih)
    nw, nh = max(1, round(iw * s)), max(1, round(ih * s))
    interp = cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR
    resized = img if (nw, nh) == (iw, ih) else cv2.resize(img, (nw, nh), interpolation=interp)
    if (nw, nh) == (w, h):
        return resized.copy() if resized is img else resized, s, (0, 0)
    canvas = np.zeros((h, w, 3), np.uint8)
    ox, oy = (w - nw) // 2, (h - nh) // 2
    canvas[oy:oy + nh, ox:ox + nw] = resized
    return canvas, s, (ox, oy)


def render_size(view: str, out_size: tuple[int, int]) -> tuple[int, int]:
    """導播畫面實際需要渲染的尺寸：split / debug 只需要較小的面板，直接渲染較省時間。"""
    w, h = out_size
    if view == "split":
        return _even(w // 2), _even(h // 2)
    if view == "debug":
        return _even(w // 4), _even(h // 4)
    return w, h


def compose_view(view: str, raw: np.ndarray, out: np.ndarray, out_size: tuple[int, int],
                 persons: list[Person], crop: np.ndarray | None, target_rect: np.ndarray | None,
                 hud: list[str] | None, debug: bool) -> np.ndarray:
    """組合最終畫面（尺寸固定為 out_size）。

    - output：導播輸出；debug 時加 HUD。
    - split ：左 = 原始畫面（debug 時含除錯圖層）、右 = 導播輸出，上方標籤 Raw / StageLens。
    - debug ：原始畫面 + 完整除錯圖層 + 右下角子畫面顯示導播輸出。
    """
    W, H = out_size
    if view == "output":
        img = out if out.shape[1::-1] == (W, H) else cv2.resize(out, (W, H))
        if debug and hud:
            draw_hud(img, hud)
        return img

    if view == "split":
        pw, ph = W // 2, H // 2
        canvas = np.zeros((H, W, 3), np.uint8)
        left, s, off = fit_into(raw, pw, ph)
        if debug:
            _draw_debug_layers(left, persons, crop, target_rect, s, off)
        right, _, _ = fit_into(out, pw, ph)
        y = (H - ph) // 2
        canvas[y:y + ph, 0:pw] = left
        canvas[y:y + ph, W - pw:W] = right
        fs = H / 1080 * 1.2
        draw_label(canvas, "Raw", (16 * fs, y - 14 * fs), (255, 255, 255), fs)
        draw_label(canvas, "StageLens", (W - pw + 16 * fs, y - 14 * fs), (255, 255, 255), fs)
        if debug and hud:
            draw_hud(canvas, hud, origin=(int(16 * fs), y + ph + int(40 * fs)))
        return canvas

    # debug
    canvas, s, off = fit_into(raw, W, H)
    _draw_debug_layers(canvas, persons, crop, target_rect, s, off)
    if hud:
        draw_hud(canvas, hud)
    pip, _, _ = fit_into(out, *render_size("debug", out_size))
    ph, pw = pip.shape[:2]
    m = max(8, W // 120)
    canvas[H - ph - m:H - m, W - pw - m:W - m] = pip
    cv2.rectangle(canvas, (W - pw - m - 1, H - ph - m - 1), (W - m, H - m), CROP_COLOR, 1)
    return canvas


def _draw_debug_layers(img, persons, crop, target_rect, scale, offset) -> None:
    t = max(1, round(img.shape[0] / 360))
    draw_persons(img, persons, scale, offset)
    if target_rect is not None:
        draw_rect(img, target_rect, TARGET_COLOR, t, dashed=True, scale=scale, offset=offset)
    if crop is not None:
        draw_rect(img, crop, CROP_COLOR, t + 1, scale=scale, offset=offset)


def _pt(p) -> tuple[int, int]:
    return int(round(p[0])), int(round(p[1]))


def _even(v: int) -> int:
    return max(2, v - v % 2)

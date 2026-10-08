import numpy as np

from src.framing.target import TargetPlanner, compute_target, target_to_rect

from .helpers import H, W, make_person


def test_no_persons_returns_none(framing_cfg):
    assert compute_target([], W, H, framing_cfg) is None


def test_invalid_persons_are_ignored(framing_cfg):
    tiny = make_person(900, 500, 0.05 * H)                 # 低於畫面高度 8%
    few = make_person(900, 300, 500)
    few.scores[:] = 0.1
    few.scores[:4] = 0.9                                   # 只有 4 個可見關鍵點
    assert compute_target([tiny, few], W, H, framing_cfg) is None


def test_single_person_padding_and_aspect(framing_cfg):
    p = make_person(cx=800, top=300, height=600)
    target = compute_target([p], W, H, framing_cfg)
    assert target is not None
    x0, y0, x1, y1 = target_to_rect(target, framing_cfg.aspect)
    kx0, ky0 = p.keypoints.min(0)
    kx1, ky1 = p.keypoints.max(0)
    h = ky1 - ky0
    # 頭頂留白 15% 身高、腳底 8%
    assert y0 <= ky0 - 0.15 * h + 1e-3
    assert y1 >= ky1 + 0.08 * h - 1e-3
    assert x0 <= kx0 and x1 >= kx1
    # 符合輸出比例，寬 = 框高 × 16/9（站立的人比較高）
    assert np.isclose((x1 - x0) / (y1 - y0), framing_cfg.aspect)
    assert np.isclose(target[2], (1.23 * h) * framing_cfg.aspect, rtol=1e-3)
    assert np.isclose(target[0], (kx0 + kx1) / 2)


def test_zoom_is_clipped(framing_cfg):
    small = make_person(900, 500, 0.1 * H)
    target = compute_target([small], W, H, framing_cfg)
    assert np.isclose(target[2], W / framing_cfg.zoom_max)
    huge = make_person(900, -200, 2 * H)
    assert np.isclose(compute_target([huge], W, H, framing_cfg)[2], W)


def test_spread_group_gives_full_shot(framing_cfg):
    group = [make_person(100, 400, 400), make_person(1820, 400, 400)]
    target = compute_target(group, W, H, framing_cfg)
    assert np.allclose(target, [W / 2, H / 2, W])


def test_burst_window_keeps_wide_box(framing_cfg):
    fps = 30
    planner = TargetPlanner(W, H, fps, framing_cfg)
    window = round(framing_cfg.burst_window_s * fps)
    jump = make_person(900, 100, 800)                      # 跳起來 / 伸展：框很大
    stand = make_person(900, 400, 400)
    wide = planner.update([jump])[2]
    for _ in range(window - 1):
        assert np.isclose(planner.update([stand])[2], wide)  # 時間窗內維持大框
    for _ in range(3):
        last = planner.update([stand])
    assert np.isclose(last[2], compute_target([stand], W, H, framing_cfg)[2])


def test_planner_bridges_short_detection_gaps(framing_cfg):
    planner = TargetPlanner(W, H, 30, framing_cfg)
    planner.update([make_person(900, 400, 400)])
    assert planner.update([]) is not None                  # 偶爾漏偵測不算跟丟
    for _ in range(30):
        last = planner.update([])
    assert last is None

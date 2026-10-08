import numpy as np
import pytest

from src.framing.camera import VirtualCamera

from .helpers import H, W


def dead_zone(cfg, w):
    return np.array([cfg.dead_xy * w, cfg.dead_xy * w / cfg.aspect, cfg.dead_zoom * w])


def assert_inside(cam, crop):
    x0, y0, x1, y1 = crop
    eps = 1e-6
    assert x0 >= -eps and y0 >= -eps and x1 <= W + eps and y1 <= H + eps
    assert W / cam.cfg.zoom_max - eps <= x1 - x0 <= W + eps


def test_static_target_converges(framing_cfg):
    cam = VirtualCamera(W, H, framing_cfg)
    target = np.array([700.0, 450.0, 900.0])
    for _ in range(400):
        cam.update(target)
    err = np.abs(cam.state - target)
    assert np.all(err <= dead_zone(framing_cfg, cam.state[2]) + 1e-6)
    assert np.allclose(cam.vel, 0)


def test_jitter_inside_dead_zone_does_not_move(framing_cfg):
    cam = VirtualCamera(W, H, framing_cfg)
    target = np.array([900.0, 500.0, 1000.0])
    cam.state[:] = target
    start = cam.state.copy()
    dz = dead_zone(framing_cfg, target[2])
    rng = np.random.default_rng(0)
    for _ in range(200):
        cam.update(target + rng.uniform(-0.5, 0.5, 3) * dz)
    assert np.array_equal(cam.state, start)


def test_teleport_respects_speed_limits(framing_cfg):
    cam = VirtualCamera(W, H, framing_cfg)
    for _ in range(300):
        cam.update(np.array([800.0, 500.0, 800.0]))
    max_pan = framing_cfg.max_pan * W
    max_zoom = framing_cfg.max_zoom * W
    prev = cam.state.copy()
    prev_vel = cam.vel.copy()
    for _ in range(300):
        cam.update(np.array([1300.0, 600.0, 1100.0]))        # 目標瞬移（不會撞到畫面邊界）
        step = cam.state - prev
        assert abs(step[0]) <= max_pan + 1e-6 and abs(step[1]) <= max_pan + 1e-6
        assert abs(step[2]) <= max_zoom + 1e-6
        # 加速度上限：速度變化不超過 max_v × accel_ratio
        dv = np.abs(cam.vel - prev_vel)
        assert dv[0] <= max_pan * framing_cfg.accel_ratio + 1e-6
        prev, prev_vel = cam.state.copy(), cam.vel.copy()
    assert abs(cam.state[0] - 1300.0) < dead_zone(framing_cfg, cam.state[2])[0] + 1


@pytest.mark.parametrize("seed", range(5))
def test_crop_always_inside_frame(framing_cfg, seed):
    cam = VirtualCamera(W, H, framing_cfg)
    rng = np.random.default_rng(seed)
    for i in range(500):
        if i % 50 < 5:
            target = None
        else:                              # 包含超出畫面、過大 / 過小的目標
            target = rng.uniform([-500, -500, 100], [W + 500, H + 500, 3 * W])
        assert_inside(cam, cam.update(target))


def test_lost_target_holds_then_returns_to_wide(framing_cfg):
    cam = VirtualCamera(W, H, framing_cfg)
    for _ in range(300):
        cam.update(np.array([600.0, 400.0, 800.0]))
    held = cam.state.copy()
    for _ in range(framing_cfg.hold_frames - 1):
        cam.update(None)
    assert np.allclose(cam.state, held, atol=1.0)          # hold 期間停住
    for _ in range(600):
        cam.update(None)
    assert cam.state[2] > W * (1 - framing_cfg.dead_zoom) - 1  # 拉回全景


def test_zoom_out_faster_than_zoom_in(framing_cfg):
    def frames_to_reach(start_w, end_w):
        cam = VirtualCamera(W, H, framing_cfg)
        cam.state[:] = [W / 2, H / 2, start_w]
        for i in range(1, 2000):
            cam.update(np.array([W / 2, H / 2, end_w]))
            if abs(cam.state[2] - end_w) <= framing_cfg.dead_zoom * cam.state[2] + 1:
                return i
        return 2000

    assert frames_to_reach(800, 1800) < frames_to_reach(1800, 800)

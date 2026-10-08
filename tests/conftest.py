from __future__ import annotations

import pytest

from src.config import Config, FramingConfig, LightingConfig

from .helpers import H, W


@pytest.fixture
def framing_cfg() -> FramingConfig:
    cfg = Config().framing
    cfg.aspect = W / H
    return cfg


@pytest.fixture
def lighting_cfg() -> LightingConfig:
    return Config().lighting

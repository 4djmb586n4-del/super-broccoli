from __future__ import annotations

import numpy as np
import pytest

from vectorizer.config import VectorizerConfig
from vectorizer.m6_regularize.angle_snap import snap_line
from vectorizer.m6_regularize.regularize import run
from vectorizer.types import Primitive, PrimitivesResult


def _line(r0, c0, r1, c1) -> Primitive:
    return Primitive(kind='line', points=np.array([[r0, c0], [r1, c1]], dtype=np.float32))


def test_snap_nearly_horizontal():
    # 1.5° off horizontal — should snap to 0°
    prim = _line(100, 0, 101, 100)   # ~0.57° — within 2° threshold
    config = VectorizerConfig(angle_snap_threshold_deg=2.0)
    snapped = snap_line(prim, config.angle_snap_threshold_deg)
    r0, c0 = snapped.points[0]
    r1, c1 = snapped.points[1]
    angle = abs(np.degrees(np.arctan2(r1 - r0, c1 - c0)))
    assert angle < 0.01


def test_snap_does_not_affect_diagonal():
    # 44° — not within 2° of 45°? Actually 44 is within 2° of 45, so it snaps.
    # Use 43° — 2° away from 45°, outside threshold
    import math
    length = 100
    r0, c0 = 100.0, 100.0
    angle_rad = math.radians(43)
    r1 = r0 + length * math.sin(angle_rad)
    c1 = c0 + length * math.cos(angle_rad)
    prim = _line(r0, c0, r1, c1)
    config = VectorizerConfig(angle_snap_threshold_deg=1.0)
    snapped = snap_line(prim, config.angle_snap_threshold_deg)
    # Should NOT snap (43° is 2° from 45° but threshold is 1°)
    dr = snapped.points[1, 0] - snapped.points[0, 0]
    dc = snapped.points[1, 1] - snapped.points[0, 1]
    snapped_angle = abs(np.degrees(np.arctan2(dr, dc)))
    assert abs(snapped_angle - 43) < 0.1


def test_run_snaps_lines_in_batch():
    prims = PrimitivesResult(primitives=[
        _line(100, 0, 100, 100),   # exact horizontal
        _line(0, 100, 100, 100),   # exact vertical
    ])
    config = VectorizerConfig(angle_snap_enabled=True)
    result = run(prims, config)
    assert len(result.primitives) == 2

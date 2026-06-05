from __future__ import annotations

import numpy as np
import pytest

from vectorizer.config import VectorizerConfig
from vectorizer.m5_primitives.lines import fit_line
from vectorizer.m5_primitives.arcs import fit_arc, arc_angular_span
from vectorizer.m5_primitives.polylines import rdp


def test_fit_line_perfect_horizontal():
    path = np.array([[10, i] for i in range(0, 50)], dtype=np.float32)
    endpoints, rms = fit_line(path)
    assert rms < 0.1
    assert abs(endpoints[0, 0] - endpoints[1, 0]) < 0.5   # same row


def test_fit_line_residual_high_for_curve():
    # Quarter circle — should NOT fit well as a line
    angles = np.linspace(0, np.pi / 2, 30)
    path = np.column_stack([50 * np.sin(angles) + 100, 50 * np.cos(angles) + 100]).astype(np.float32)
    _, rms = fit_line(path)
    assert rms > 3.0


def test_fit_arc_circle():
    angles = np.linspace(0, 2 * np.pi, 60, endpoint=False)
    r = 40.0
    path = np.column_stack([100 + r * np.sin(angles), 150 + r * np.cos(angles)]).astype(np.float32)
    center, radius, _, _, rms = fit_arc(path)
    assert abs(radius - r) < 1.0
    assert rms < 0.5


def test_shallow_arc_short_sweep():
    """A shallow open arc must report a small signed sweep, not ~360°.

    Regression: shallow rooflines were rendered the long way around as
    near-complete circles because the sweep/large-arc flags were wrong.
    """
    from vectorizer.m5_primitives.arcs import fit_arc
    # 40° arc of a large-radius circle
    angles = np.linspace(np.radians(70), np.radians(110), 40)
    r = 700.0
    path = np.column_stack([200 + r * np.sin(angles), 400 + r * np.cos(angles)]).astype(np.float32)
    _, _, start_a, end_a, _ = fit_arc(path)
    sweep = abs(end_a - start_a)
    assert sweep < 90        # small arc, NOT a near-full circle
    assert sweep > 20


def test_rdp_reduces_points():
    path = np.array([[i, i] for i in range(100)], dtype=np.float32)  # diagonal line
    simplified = rdp(path, epsilon=1.0)
    assert len(simplified) == 2   # perfect diagonal → 2 points


def test_rdp_preserves_corners():
    # L-shape: right then up
    path = np.array([[0, i] for i in range(50)] + [[j, 49] for j in range(1, 50)], dtype=np.float32)
    simplified = rdp(path, epsilon=0.5)
    assert len(simplified) >= 3   # at least start, corner, end

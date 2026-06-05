from __future__ import annotations

import numpy as np
import pytest

from vectorizer.config import VectorizerConfig
from vectorizer.m3_skeleton.skeleton import run
from vectorizer.m3_skeleton.thin import thin
from vectorizer.m3_skeleton.prune import prune_spurs


def test_thin_thick_bar_becomes_single_line():
    """A thick horizontal bar should thin to a single-pixel-wide skeleton."""
    img = np.zeros((50, 200), dtype=np.uint8)
    img[20:31, 10:190] = 255   # 11-pixel-tall bar
    skel = thin(img)
    # All skeleton pixels should be at a single row (within ±2)
    rows = np.unique(np.where(skel)[0])
    assert len(rows) <= 3


def test_skeleton_run_returns_bool(horizontal_line_image):
    config = VectorizerConfig(spur_min_length=0)
    result = run(horizontal_line_image, config)
    assert result.skeleton.dtype == bool
    assert result.distance.dtype == np.float32


def test_prune_removes_short_spurs():
    """An isolated short branch should be pruned."""
    skel = np.zeros((30, 30), dtype=bool)
    # Horizontal line
    skel[15, 5:25] = True
    # Short upward spur of length 3 at col 15
    skel[12:15, 15] = True
    pruned = prune_spurs(skel, min_length=5)
    assert not pruned[12, 15]   # spur removed


def test_prune_keeps_long_branches():
    skel = np.zeros((50, 50), dtype=bool)
    skel[25, 5:45] = True        # long horizontal
    skel[5:25, 25] = True        # long vertical (20px) — should survive
    pruned = prune_spurs(skel, min_length=5)
    # The vertical branch should mostly survive
    assert pruned[5:20, 25].any()

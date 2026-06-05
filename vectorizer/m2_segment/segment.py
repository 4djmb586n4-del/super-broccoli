"""M2 Layer segmentation — Phase 1 stub.

In Phase 0, all pixels are treated as a single 'geometry' layer.
Phase 1 will implement classical segmentation (CC analysis, Hough/FFT hatching detection).
Phase 3 will add an ML variant (U-Net / DeepLabv3+).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from vectorizer.config import VectorizerConfig


@dataclass
class SegmentResult:
    # Each mask is a uint8 binary image (0/255), same shape as input
    geometry: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.uint8))
    text: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.uint8))
    hatching: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.uint8))
    dimensions: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.uint8))
    symbols: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.uint8))


def run(binary: np.ndarray, config: VectorizerConfig) -> SegmentResult:
    """Phase 0 stub: pass all pixels through as geometry."""
    return SegmentResult(geometry=binary.copy())

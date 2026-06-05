from __future__ import annotations

import numpy as np

from vectorizer.config import VectorizerConfig
from vectorizer.types import PreprocessResult

from .binarize import binarize, to_grayscale
from .denoise import denoise
from .deskew import deskew, estimate_skew_angle


def run(image: np.ndarray, config: VectorizerConfig) -> PreprocessResult:
    gray = to_grayscale(image)

    angle = 0.0
    if config.deskew_enabled:
        # Quick pre-binarise with Otsu to get clean edges for angle estimation
        pre_binary = binarize(gray, method='otsu')
        angle = estimate_skew_angle(pre_binary, max_angle=config.deskew_max_angle)
        gray = deskew(gray, angle)

    binary = binarize(gray, method=config.binarize_method, window=config.binarize_window)
    binary = denoise(binary, median_ksize=config.denoise_median_ksize, min_cc_area=config.min_cc_area)

    return PreprocessResult(binary=binary, deskew_angle=angle)

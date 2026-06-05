from __future__ import annotations

import numpy as np
import pytest

from vectorizer.config import VectorizerConfig
from vectorizer.m1_preprocess.binarize import binarize, to_grayscale
from vectorizer.m1_preprocess.denoise import denoise
from vectorizer.m1_preprocess.deskew import estimate_skew_angle
from vectorizer.m1_preprocess.preprocess import run


def test_to_grayscale_already_gray():
    gray = np.full((50, 50), 128, dtype=np.uint8)
    result = to_grayscale(gray)
    assert result.shape == (50, 50)
    assert result.dtype == np.uint8


def test_to_grayscale_bgr():
    bgr = np.zeros((50, 50, 3), dtype=np.uint8)
    bgr[:, :, 0] = 200  # blue channel
    result = to_grayscale(bgr)
    assert result.ndim == 2


def test_binarize_otsu_produces_binary():
    gray = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
    binary = binarize(gray, method='otsu')
    unique = np.unique(binary)
    assert set(unique).issubset({0, 255})


def test_binarize_adaptive_produces_binary():
    gray = np.random.randint(100, 200, (100, 100), dtype=np.uint8)
    binary = binarize(gray, method='adaptive', window=31)
    unique = np.unique(binary)
    assert set(unique).issubset({0, 255})


def test_denoise_removes_tiny_components():
    img = np.zeros((100, 100), dtype=np.uint8)
    img[10, 10] = 255   # single pixel noise
    img[20:30, 20:80] = 255   # large feature → kept
    result = denoise(img, median_ksize=1, min_cc_area=10)
    assert result[10, 10] == 0          # noise removed
    assert result[25, 50] == 255        # feature preserved


def test_preprocess_run_returns_correct_shape(horizontal_line_image):
    config = VectorizerConfig(deskew_enabled=False)
    result = run(horizontal_line_image, config)
    assert result.binary.shape == horizontal_line_image.shape
    assert set(np.unique(result.binary)).issubset({0, 255})


def test_deskew_angle_near_zero_for_horizontal(horizontal_line_image):
    from vectorizer.m1_preprocess.binarize import binarize
    binary = binarize(horizontal_line_image, 'otsu')
    angle = estimate_skew_angle(binary, max_angle=10.0)
    assert abs(angle) < 1.0  # should detect ≈0° skew

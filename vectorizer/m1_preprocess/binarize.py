from __future__ import annotations

import cv2
import numpy as np
from skimage.filters import threshold_sauvola


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert any image to uint8 grayscale."""
    if image.ndim == 2:
        gray = image
    elif image.shape[2] == 4:
        gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if gray.dtype != np.uint8:
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    return gray


def binarize_otsu(gray: np.ndarray) -> np.ndarray:
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return binary


def binarize_sauvola(gray: np.ndarray, window: int = 51) -> np.ndarray:
    window = window if window % 2 == 1 else window + 1
    thresh = threshold_sauvola(gray, window_size=window)
    binary = (gray < thresh).astype(np.uint8) * 255
    return binary


def binarize_adaptive(gray: np.ndarray, window: int = 51) -> np.ndarray:
    window = window if window % 2 == 1 else window + 1
    binary = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        window,
        10,
    )
    return binary


def binarize(gray: np.ndarray, method: str = 'otsu', window: int = 51) -> np.ndarray:
    if method == 'sauvola':
        return binarize_sauvola(gray, window)
    if method == 'adaptive':
        return binarize_adaptive(gray, window)
    return binarize_otsu(gray)

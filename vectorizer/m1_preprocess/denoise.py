from __future__ import annotations

import cv2
import numpy as np


def denoise(binary: np.ndarray, median_ksize: int = 3, min_cc_area: int = 10) -> np.ndarray:
    """Remove noise from a binary (0/255) image.

    1. Median blur to kill isolated salt-and-pepper pixels.
    2. Remove connected components smaller than *min_cc_area* pixels.
    """
    result = binary.copy()

    if median_ksize > 1:
        ksize = median_ksize if median_ksize % 2 == 1 else median_ksize + 1
        result = cv2.medianBlur(result, ksize)

    if min_cc_area > 1:
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(result, connectivity=8)
        for label in range(1, num_labels):
            if stats[label, cv2.CC_STAT_AREA] < min_cc_area:
                result[labels == label] = 0

    return result

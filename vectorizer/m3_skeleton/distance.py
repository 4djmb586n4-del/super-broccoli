from __future__ import annotations

import numpy as np
from scipy.ndimage import distance_transform_edt


def distance_map(binary: np.ndarray, skeleton: np.ndarray) -> np.ndarray:
    """Return distance-transform values sampled at skeleton pixels.

    *binary* is uint8 0/255; *skeleton* is bool.
    The returned array has the same shape; non-skeleton pixels are 0.
    """
    dist = distance_transform_edt(binary > 0).astype(np.float32)
    result = np.zeros_like(dist)
    result[skeleton] = dist[skeleton]
    return result

from __future__ import annotations

import numpy as np
from skimage.morphology import thin as skimage_thin


def thin(binary: np.ndarray) -> np.ndarray:
    """Zhang-Suen thinning via skimage.

    *binary* is uint8 0/255. Returns bool array of the same shape.
    """
    return skimage_thin(binary > 0)

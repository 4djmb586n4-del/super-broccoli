from __future__ import annotations

import numpy as np
from scipy.ndimage import label


def _neighbour_count(skel: np.ndarray) -> np.ndarray:
    """Return an array counting 8-connected True neighbours for each pixel."""
    kernel = np.ones((3, 3), dtype=np.uint8)
    from scipy.ndimage import convolve
    counts = convolve(skel.astype(np.uint8), kernel, mode='constant', cval=0)
    counts[~skel] = 0
    counts[skel] -= 1  # exclude the pixel itself
    return counts


def prune_spurs(skeleton: np.ndarray, min_length: int) -> np.ndarray:
    """Remove short spur branches from *skeleton*.

    Iteratively removes endpoint pixels that belong to branches shorter than
    *min_length* pixels and connect to a junction (degree ≥ 3).
    """
    if min_length <= 0:
        return skeleton

    skel = skeleton.copy()

    for _ in range(min_length):
        nc = _neighbour_count(skel)
        # Endpoint pixels: exactly 1 neighbour
        endpoints = skel & (nc == 1)
        if not endpoints.any():
            break
        skel[endpoints] = False

    return skel

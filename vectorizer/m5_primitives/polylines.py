from __future__ import annotations

import numpy as np


def rdp(points: np.ndarray, epsilon: float) -> np.ndarray:
    """Ramer-Douglas-Peucker simplification.

    *points* is (N, 2); returns simplified (M, 2) with M ≤ N.
    """
    if len(points) < 3:
        return points

    start, end = points[0], points[-1]
    seg = end - start
    seg_len = np.linalg.norm(seg)

    if seg_len < 1e-9:
        dists = np.linalg.norm(points - start, axis=1)
    else:
        seg_unit = seg / seg_len
        vecs = points - start
        t = vecs @ seg_unit
        proj = start + t[:, None] * seg_unit
        dists = np.linalg.norm(points - proj, axis=1)

    max_idx = int(np.argmax(dists))
    max_dist = dists[max_idx]

    if max_dist > epsilon:
        left  = rdp(points[: max_idx + 1], epsilon)
        right = rdp(points[max_idx:], epsilon)
        return np.vstack([left[:-1], right])

    return np.array([start, end])

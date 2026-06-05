from __future__ import annotations

import numpy as np


def fit_line(path: np.ndarray) -> tuple[np.ndarray, float]:
    """Total-least-squares line fit.

    Returns (endpoints_2x2, rms_residual).
    endpoints[0] and endpoints[1] are the projections of the first and last
    path points onto the fitted line, in (row, col) pixel space.
    """
    if len(path) < 2:
        return path[:2].copy() if len(path) == 2 else np.array([path[0], path[0]]), 0.0

    pts = path.astype(np.float64)
    centroid = pts.mean(axis=0)
    centered = pts - centroid

    # SVD: right singular vectors give principal directions
    _, _, Vt = np.linalg.svd(centered, full_matrices=False)
    direction = Vt[0]  # unit vector along the line

    # Project all points onto the line
    projections = centered @ direction  # scalar distances along line

    # Project first/last path points (not min/max) to preserve ordering
    t_start = (path[0].astype(np.float64) - centroid) @ direction
    t_end   = (path[-1].astype(np.float64) - centroid) @ direction

    p_start = centroid + t_start * direction
    p_end   = centroid + t_end   * direction

    # RMS residual: distance of each point from the line
    residuals = centered - projections[:, None] * direction
    rms = float(np.sqrt((residuals ** 2).sum(axis=1).mean()))

    endpoints = np.stack([p_start, p_end]).astype(np.float32)
    return endpoints, rms

from __future__ import annotations

import numpy as np


def _taubin_fit(pts: np.ndarray) -> tuple[float, float, float]:
    """Taubin algebraic circle fit.

    Returns (center_row, center_col, radius).
    Raises ValueError if the fit is degenerate.
    """
    x = pts[:, 1].astype(np.float64)  # col
    y = pts[:, 0].astype(np.float64)  # row

    n = len(x)
    if n < 3:
        raise ValueError("Need at least 3 points for circle fit")

    # Taubin method
    x_ = x - x.mean()
    y_ = y - y.mean()

    Mxx = (x_ * x_).mean()
    Myy = (y_ * y_).mean()
    Mxy = (x_ * y_).mean()
    Mxz = (x_ * (x_ ** 2 + y_ ** 2)).mean()
    Myz = (y_ * (x_ ** 2 + y_ ** 2)).mean()
    Mzz = ((x_ ** 2 + y_ ** 2) ** 2).mean()

    Mz = Mxx + Myy
    Cov_xy = Mxx * Myy - Mxy ** 2
    Var_z = Mzz - Mz ** 2

    A2 = 4 * Cov_xy - 3 * Mz ** 2 - Mzz
    A1 = Var_z * Mz + 4 * Cov_xy * Mz - Mxz ** 2 - Myz ** 2
    A0 = Mxz * (Mxz * Myy - Myz * Mxy) + Myz * (Myz * Mxx - Mxz * Mxy) - Var_z * Cov_xy
    A22 = 2 * A2

    # Newton's method to find the smallest non-negative root
    x0 = 0.0
    for _ in range(99):
        dx = (A0 + x0 * (A1 + x0 * A2)) / (A1 + x0 * (A22 + 3 * x0 * A2))
        x0 -= dx
        if abs(dx) < 1e-12:
            break

    DET = x0 ** 2 - x0 * Mz + Cov_xy
    if abs(DET) < 1e-10:
        raise ValueError("Degenerate Taubin fit")

    center_x = (Mxz * (Myy - x0) - Myz * Mxy) / (2 * DET) + x.mean()
    center_y = (Myz * (Mxx - x0) - Mxz * Mxy) / (2 * DET) + y.mean()
    # Compute radius as mean distance from fitted centre (robust to numerical issues)
    radius = float(np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2).mean())

    return float(center_y), float(center_x), float(radius)  # (row, col, r)


def fit_arc(path: np.ndarray) -> tuple[tuple[float, float], float, float, float, float]:
    """Fit a circular arc to *path* (Nx2 in row/col).

    Returns (center_rc, radius, start_angle_deg, end_angle_deg, rms_residual).
    Raises ValueError on failure.
    """
    cr, cc, r = _taubin_fit(path)

    # Residual
    dist = np.sqrt((path[:, 0] - cr) ** 2 + (path[:, 1] - cc) ** 2)
    rms = float(np.sqrt(((dist - r) ** 2).mean()))

    # Start angle from the first path point (atan2(row-cr, col-cc) → matches
    # the col=x, row=y convention used by the SVG builder).
    start_angle = float(np.degrees(np.arctan2(path[0, 0] - cr, path[0, 1] - cc)))
    # Signed sweep traversed along the path (sign encodes draw direction).
    signed_sweep = signed_arc_span(path, (cr, cc))
    end_angle = start_angle + signed_sweep

    return (cr, cc), r, start_angle, end_angle, rms


def signed_arc_span(path: np.ndarray, center: tuple[float, float]) -> float:
    """Signed total angular sweep (degrees) traversed along the path.

    Positive = counter-clockwise in (row, col) space. Sign and magnitude come
    from summing consecutive angular steps, so an open 46° arc yields ±46°,
    not 314° — which is what fixes arcs being rendered the long way around.
    """
    cr, cc = center
    vecs = path.astype(np.float64) - np.array([cr, cc])
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms = np.where(norms < 1e-9, 1.0, norms)
    unit = vecs / norms
    cross = unit[:-1, 0] * unit[1:, 1] - unit[:-1, 1] * unit[1:, 0]
    dot   = (unit[:-1] * unit[1:]).sum(axis=1)
    steps = np.arctan2(cross, dot)
    return float(np.degrees(steps.sum()))


def arc_angular_span(path: np.ndarray, center: tuple[float, float]) -> float:
    """Return the total angular span of path points around center (degrees).

    Uses cumulative angular step sum instead of unwrap to handle any path ordering.
    """
    cr, cc = center
    vecs = path.astype(np.float64) - np.array([cr, cc])
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms = np.where(norms < 1e-9, 1.0, norms)
    unit = vecs / norms

    # Cross and dot between consecutive unit vectors → signed angular steps
    cross = unit[:-1, 0] * unit[1:, 1] - unit[:-1, 1] * unit[1:, 0]
    dot   = (unit[:-1] * unit[1:]).sum(axis=1)
    steps = np.arctan2(cross, dot)
    return float(abs(np.degrees(steps.sum())))


def endpoints_distance(path: np.ndarray) -> float:
    """Euclidean distance between the first and last path points."""
    return float(np.linalg.norm(path[-1].astype(np.float64) - path[0].astype(np.float64)))

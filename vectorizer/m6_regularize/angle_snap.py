from __future__ import annotations

import numpy as np

from vectorizer.types import Primitive

# Canonical angles to snap to (degrees)
_SNAP_TARGETS = np.array([0, 45, 90, 135, 180, 225, 270, 315, 360], dtype=np.float64)


def _nearest_snap(angle_deg: float, threshold: float) -> float | None:
    """Return the nearest snap angle if within threshold, else None."""
    a = angle_deg % 360
    diffs = np.abs(_SNAP_TARGETS - a)
    # Also check wrap-around
    diffs = np.minimum(diffs, 360 - diffs)
    idx = int(np.argmin(diffs))
    if diffs[idx] <= threshold:
        return float(_SNAP_TARGETS[idx] % 180)  # normalise to [0,180)
    return None


def snap_line(prim: Primitive, threshold_deg: float) -> Primitive:
    """Snap a line primitive to the nearest cardinal/diagonal angle."""
    if prim.kind != 'line' or len(prim.points) < 2:
        return prim

    p0, p1 = prim.points[0].astype(np.float64), prim.points[1].astype(np.float64)
    dr, dc = p1[0] - p0[0], p1[1] - p0[1]
    # Angle from positive col-axis (horizontal), matching arctan2(row_delta, col_delta)
    current_angle = float(np.degrees(np.arctan2(dr, dc))) % 360

    target = _nearest_snap(current_angle, threshold_deg)
    if target is None:
        return prim

    # Rotate both endpoints around the midpoint to the snapped angle
    mid = (p0 + p1) / 2.0
    half_length = np.linalg.norm(p1 - p0) / 2.0
    target_rad = np.radians(target)
    direction = np.array([np.sin(target_rad), np.cos(target_rad)])  # (row, col)
    # Match original direction sign
    if np.dot(np.array([dr, dc]), direction) < 0:
        direction = -direction

    new_p0 = mid - half_length * direction
    new_p1 = mid + half_length * direction

    new_prim = Primitive(
        kind='line',
        layer=prim.layer,
        points=np.stack([new_p0, new_p1]).astype(np.float32),
        stroke_width=prim.stroke_width,
    )
    return new_prim

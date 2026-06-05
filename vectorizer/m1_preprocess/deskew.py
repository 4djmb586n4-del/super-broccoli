from __future__ import annotations

import cv2
import numpy as np


def estimate_skew_angle(binary: np.ndarray, max_angle: float = 10.0) -> float:
    """Return the dominant skew angle (degrees) of the document.

    Uses Hough line transform on the edge image. Returns 0.0 if no reliable
    angle is found or the detected angle exceeds max_angle (likely not skew).
    """
    edges = cv2.Canny(binary, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=100,
        minLineLength=binary.shape[1] // 10,
        maxLineGap=20,
    )
    if lines is None:
        return 0.0

    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        dx, dy = x2 - x1, y2 - y1
        if dx == 0:
            continue
        angle = np.degrees(np.arctan2(dy, dx))
        # Normalise to [-45, 45] (angles near ±90 are vertical lines → skip)
        if abs(angle) > 45:
            angle = angle - np.sign(angle) * 90
        angles.append(angle)

    if not angles:
        return 0.0

    median_angle = float(np.median(angles))
    if abs(median_angle) > max_angle:
        return 0.0
    return median_angle


def deskew(binary: np.ndarray, angle: float) -> np.ndarray:
    """Rotate *binary* by -*angle* degrees around its centre (white background)."""
    if abs(angle) < 0.05:
        return binary
    h, w = binary.shape[:2]
    cx, cy = w / 2.0, h / 2.0
    M = cv2.getRotationMatrix2D((cx, cy), -angle, 1.0)
    rotated = cv2.warpAffine(
        binary,
        M,
        (w, h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return rotated

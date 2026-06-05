from __future__ import annotations

import numpy as np
import cv2
import pytest


def make_blank(h: int = 200, w: int = 300) -> np.ndarray:
    """White canvas (like paper), uint8."""
    return np.full((h, w), 255, dtype=np.uint8)


def draw_line(canvas: np.ndarray, p0: tuple, p1: tuple, thickness: int = 3) -> np.ndarray:
    """Draw a black line on a white canvas, matching real scanned drawings."""
    img = canvas.copy()
    cv2.line(img, (p0[1], p0[0]), (p1[1], p1[0]), 0, thickness)
    return img


def draw_circle_img(canvas: np.ndarray, center: tuple, radius: int, thickness: int = 3) -> np.ndarray:
    img = canvas.copy()
    cv2.circle(img, (center[1], center[0]), radius, 0, thickness)
    return img


@pytest.fixture
def horizontal_line_image():
    """200×300 image with a single horizontal line at row 100."""
    canvas = make_blank(200, 300)
    return draw_line(canvas, (100, 30), (100, 270), thickness=5)


@pytest.fixture
def diagonal_line_image():
    """200×300 image with a ~1.5° skewed near-horizontal line."""
    canvas = make_blank(200, 300)
    # Slight skew: start at (100, 30), end at (105, 270) — ~1.4°
    return draw_line(canvas, (100, 30), (105, 270), thickness=5)


@pytest.fixture
def circle_image():
    """200×300 image with a circle."""
    canvas = make_blank(200, 300)
    return draw_circle_img(canvas, center=(100, 150), radius=50, thickness=4)


@pytest.fixture
def floor_plan_image():
    """Simple synthetic floor plan: outer rectangle + interior wall + door arc."""
    canvas = make_blank(400, 500)
    # Outer walls
    cv2.rectangle(canvas, (20, 20), (480, 380), 0, 5)
    # Interior wall
    cv2.line(canvas, (250, 20), (250, 200), 0, 4)
    # Door arc
    cv2.ellipse(canvas, (250, 200), (60, 60), 0, 0, 90, 0, 3)
    return canvas

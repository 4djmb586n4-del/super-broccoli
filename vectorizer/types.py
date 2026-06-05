from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

import networkx as nx
import numpy as np


@dataclass
class PreprocessResult:
    binary: np.ndarray          # uint8 (H, W), values 0 or 255
    deskew_angle: float = 0.0   # degrees applied; 0 if deskew disabled
    scale_factor: float = 1.0   # reserved for future DPI normalisation


@dataclass
class SkeletonResult:
    skeleton: np.ndarray        # bool (H, W)
    distance: np.ndarray        # float32 (H, W) distance-transform values at skeleton pixels


@dataclass
class GraphResult:
    graph: nx.Graph             # nodes: (row, col) tuples; edges have attr 'path': ndarray (N,2)


@dataclass
class Primitive:
    kind: Literal['line', 'arc', 'circle', 'polyline']
    layer: str = 'geometry'
    # endpoints / control points in pixel (row, col) space
    points: np.ndarray = field(default_factory=lambda: np.empty((0, 2), dtype=np.float32))
    # arc / circle only
    center: Optional[tuple[float, float]] = None   # (row, col)
    radius: Optional[float] = None
    start_angle: Optional[float] = None            # degrees
    end_angle: Optional[float] = None              # degrees
    # metadata
    stroke_width: float = 1.0   # reconstructed from distance map (pixels)


@dataclass
class PrimitivesResult:
    primitives: list[Primitive] = field(default_factory=list)


@dataclass
class SVGResult:
    svg_string: str
    viewbox: tuple[float, float, float, float]  # (min_x, min_y, width, height)

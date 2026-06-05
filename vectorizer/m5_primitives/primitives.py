from __future__ import annotations

import numpy as np
import networkx as nx

from vectorizer.config import VectorizerConfig
from vectorizer.types import GraphResult, Primitive, PrimitivesResult

from .arcs import arc_angular_span, endpoints_distance, fit_arc
from .lines import fit_line
from .polylines import rdp


def _classify_edge(path: np.ndarray, config: VectorizerConfig) -> Primitive:
    if len(path) < 2:
        return Primitive(kind='line', points=path[:2] if len(path) >= 2 else np.zeros((2, 2), np.float32))

    # ── Try line ─────────────────────────────────────────────────────────────
    endpoints, line_rms = fit_line(path)
    if line_rms <= config.line_residual:
        return Primitive(kind='line', points=endpoints)

    # ── Try arc / circle ─────────────────────────────────────────────────────
    if len(path) >= 5:
        try:
            center, radius, start_a, end_a, arc_rms = fit_arc(path)
            if (
                arc_rms <= config.arc_residual
                and config.min_arc_radius <= radius <= config.max_arc_radius
            ):
                span = arc_angular_span(path, center)
                # A closed circle requires the path to actually loop back on itself;
                # guard against winding open curves being misclassified as circles.
                endpoint_gap = endpoints_distance(path)
                is_closed = span >= 340 and endpoint_gap < radius * 0.3
                if is_closed:
                    return Primitive(
                        kind='circle',
                        points=np.array([list(center)], dtype=np.float32),
                        center=center,
                        radius=radius,
                    )
                return Primitive(
                    kind='arc',
                    points=np.stack([path[0], path[-1]]).astype(np.float32),
                    center=center,
                    radius=radius,
                    start_angle=start_a,
                    end_angle=end_a,
                )
        except (ValueError, np.linalg.LinAlgError):
            pass

    # ── Fallback: polyline ────────────────────────────────────────────────────
    simplified = rdp(path, epsilon=config.dp_epsilon)
    return Primitive(kind='polyline', points=simplified.astype(np.float32))


def run(graph_result: GraphResult, config: VectorizerConfig) -> PrimitivesResult:
    primitives: list[Primitive] = []

    for u, v, data in graph_result.graph.edges(data=True):
        path: np.ndarray = data.get('path', np.array([list(u), list(v)], dtype=np.float32))
        length: float = data.get('length', float(np.linalg.norm(
            np.array(v, dtype=float) - np.array(u, dtype=float)
        )))

        if length < config.min_primitive_length:
            continue

        prim = _classify_edge(path, config)
        primitives.append(prim)

    return PrimitivesResult(primitives=primitives)

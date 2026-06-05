from __future__ import annotations

from vectorizer.config import VectorizerConfig
from vectorizer.types import Primitive, PrimitivesResult

from .angle_snap import snap_line


def run(prims_result: PrimitivesResult, config: VectorizerConfig) -> PrimitivesResult:
    primitives = prims_result.primitives

    if config.angle_snap_enabled:
        primitives = [
            snap_line(p, config.angle_snap_threshold_deg) if p.kind == 'line' else p
            for p in primitives
        ]

    # Phase 1 operations (node_weld, collinear merge, corner cleanup) will be
    # added here when fidelity_level parameter is wired.

    return PrimitivesResult(primitives=primitives)

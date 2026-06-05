from __future__ import annotations

from typing import Literal

from pydantic_settings import BaseSettings


class VectorizerConfig(BaseSettings):
    # ── M1 preprocessing ─────────────────────────────────────────────────────
    deskew_enabled: bool = True
    deskew_max_angle: float = 10.0          # degrees; larger rotations ignored (likely not skew)
    binarize_method: Literal['otsu', 'sauvola', 'adaptive'] = 'otsu'
    binarize_window: int = 51               # Sauvola / adaptive window size (must be odd)
    denoise_median_ksize: int = 3           # 0 = disabled
    min_cc_area: int = 10                   # connected-component despeckle threshold (px²)

    # ── M3 skeletonisation ────────────────────────────────────────────────────
    spur_min_length: int = 10               # prune branches shorter than this (px)

    # ── M5 primitive fitting ──────────────────────────────────────────────────
    line_residual: float = 1.5             # max RMS (px) to accept TLS line fit
    arc_residual: float = 1.5              # max RMS (px) to accept Taubin arc fit
    min_arc_radius: float = 3.0            # arcs with r < this are skipped
    max_arc_radius: float = 5000.0         # arcs with r > this treated as lines
    dp_epsilon: float = 1.0               # Douglas-Peucker tolerance (px)
    min_primitive_length: float = 4.0     # edges shorter than this are dropped

    # ── M6 regularisation ────────────────────────────────────────────────────
    angle_snap_enabled: bool = True
    angle_snap_threshold_deg: float = 2.0  # snap if within this of 0/45/90/135°
    node_weld_epsilon: float = 2.0         # Phase 1: merge nodes within this distance (px)
    fidelity_level: float = 0.5            # 0 = maximum clean, 1 = maximum faithful

    # ── M9 SVG output ─────────────────────────────────────────────────────────
    output_dpi: int = 96                   # used to scale pixel coords to SVG units
    svg_default_stroke_width: float = 0.5
    svg_decimal_places: int = 2

    model_config = {"env_prefix": "VECTORIZER_", "case_sensitive": False}

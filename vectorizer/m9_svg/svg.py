from __future__ import annotations

import numpy as np

from vectorizer.config import VectorizerConfig
from vectorizer.types import PrimitivesResult, SVGResult

from .builder import build_svg


def run(
    prims_result: PrimitivesResult,
    config: VectorizerConfig,
    image_shape: tuple[int, int] | None = None,
) -> SVGResult:
    return build_svg(prims_result, config, image_shape=image_shape)

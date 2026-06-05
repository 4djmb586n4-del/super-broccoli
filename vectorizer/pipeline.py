from __future__ import annotations

import numpy as np

from vectorizer import m1_preprocess as m1
from vectorizer import m3_skeleton as m3
from vectorizer import m4_graph as m4
from vectorizer import m5_primitives as m5
from vectorizer import m6_regularize as m6
from vectorizer import m9_svg as m9
from vectorizer.config import VectorizerConfig
from vectorizer.types import SVGResult


def run(image: np.ndarray, config: VectorizerConfig | None = None) -> SVGResult:
    if config is None:
        config = VectorizerConfig()

    pre    = m1.run(image, config)
    skel   = m3.run(pre.binary, config)
    graph  = m4.run(skel.skeleton, config)
    prims  = m5.run(graph, config)
    prims  = m6.run(prims, config)
    result = m9.run(prims, config, image_shape=pre.binary.shape[:2])

    return result

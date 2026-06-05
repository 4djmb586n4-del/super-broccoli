from __future__ import annotations

import numpy as np

from vectorizer.config import VectorizerConfig
from vectorizer.types import GraphResult

from .build import build_graph


def run(skeleton: np.ndarray, config: VectorizerConfig) -> GraphResult:
    G = build_graph(skeleton)
    return GraphResult(graph=G)

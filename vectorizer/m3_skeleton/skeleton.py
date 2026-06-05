from __future__ import annotations

import numpy as np

from vectorizer.config import VectorizerConfig
from vectorizer.types import SkeletonResult

from .distance import distance_map
from .prune import prune_spurs
from .thin import thin


def run(binary: np.ndarray, config: VectorizerConfig) -> SkeletonResult:
    skel = thin(binary)
    skel = prune_spurs(skel, min_length=config.spur_min_length)
    dist = distance_map(binary, skel)
    return SkeletonResult(skeleton=skel, distance=dist)

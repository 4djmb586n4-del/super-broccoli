"""M8 Symbol recognition — Phase 4 stub."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from vectorizer.config import VectorizerConfig


@dataclass
class SymbolInstance:
    symbol_id: str
    row: float
    col: float
    scale: float = 1.0
    angle_deg: float = 0.0


@dataclass
class SymbolResult:
    instances: list[SymbolInstance] = field(default_factory=list)


def run(symbol_mask: np.ndarray, config: VectorizerConfig) -> SymbolResult:
    """Phase 4 stub: returns empty symbol result."""
    return SymbolResult()

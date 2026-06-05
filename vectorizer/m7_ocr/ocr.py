"""M7 OCR — Phase 2 stub."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from vectorizer.config import VectorizerConfig


@dataclass
class TextElement:
    text: str
    row: float
    col: float
    width: float
    height: float
    angle_deg: float = 0.0


@dataclass
class OCRResult:
    elements: list[TextElement] = field(default_factory=list)


def run(text_mask: np.ndarray, config: VectorizerConfig) -> OCRResult:
    """Phase 2 stub: returns empty OCR result."""
    return OCRResult()

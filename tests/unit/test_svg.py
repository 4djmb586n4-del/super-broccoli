from __future__ import annotations

import numpy as np
import pytest

from vectorizer.config import VectorizerConfig
from vectorizer.m9_svg.svg import run
from vectorizer.types import Primitive, PrimitivesResult


def _line_prim(r0, c0, r1, c1) -> Primitive:
    return Primitive(kind='line', points=np.array([[r0, c0], [r1, c1]], dtype=np.float32))


def _circle_prim(cr, cc, r) -> Primitive:
    return Primitive(
        kind='circle',
        points=np.array([[cr, cc]], dtype=np.float32),
        center=(float(cr), float(cc)),
        radius=float(r),
    )


def test_svg_contains_line_element():
    prims = PrimitivesResult(primitives=[_line_prim(10, 20, 10, 80)])
    config = VectorizerConfig()
    result = run(prims, config)
    assert '<line' in result.svg_string


def test_svg_contains_circle_element():
    prims = PrimitivesResult(primitives=[_circle_prim(100, 150, 40)])
    config = VectorizerConfig()
    result = run(prims, config)
    assert '<circle' in result.svg_string


def test_svg_is_valid_xml():
    from xml.etree import ElementTree as ET
    prims = PrimitivesResult(primitives=[_line_prim(0, 0, 50, 50)])
    config = VectorizerConfig()
    result = run(prims, config)
    ET.fromstring(result.svg_string)   # raises if invalid


def test_svg_empty_primitives():
    prims = PrimitivesResult(primitives=[])
    config = VectorizerConfig()
    result = run(prims, config, image_shape=(200, 300))
    assert result.svg_string.startswith('<?xml') or '<svg' in result.svg_string


def test_svg_has_layer_groups():
    prims = PrimitivesResult(primitives=[_line_prim(0, 0, 10, 10)])
    config = VectorizerConfig()
    result = run(prims, config)
    assert 'layer-geometry' in result.svg_string

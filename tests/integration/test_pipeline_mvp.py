from __future__ import annotations

from xml.etree import ElementTree as ET

import numpy as np
import pytest

from vectorizer.config import VectorizerConfig
from vectorizer.pipeline import run as pipeline_run


def test_pipeline_horizontal_line(horizontal_line_image):
    config = VectorizerConfig(deskew_enabled=False, spur_min_length=5)
    result = pipeline_run(horizontal_line_image, config)
    assert '<svg' in result.svg_string
    ET.fromstring(result.svg_string)


def test_pipeline_circle(circle_image):
    config = VectorizerConfig(deskew_enabled=False)
    result = pipeline_run(circle_image, config)
    assert '<svg' in result.svg_string
    # Should contain at least one geometric element
    assert '<circle' in result.svg_string or '<path' in result.svg_string or '<line' in result.svg_string


def test_pipeline_floor_plan(floor_plan_image):
    config = VectorizerConfig(deskew_enabled=False, spur_min_length=8)
    result = pipeline_run(floor_plan_image, config)
    root = ET.fromstring(result.svg_string)
    # Count geometric elements
    ns = {'svg': 'http://www.w3.org/2000/svg'}
    elements = root.findall('.//{http://www.w3.org/2000/svg}line') + \
               root.findall('.//{http://www.w3.org/2000/svg}path') + \
               root.findall('.//{http://www.w3.org/2000/svg}circle') + \
               root.findall('.//{http://www.w3.org/2000/svg}polyline')
    assert len(elements) >= 1


def test_pipeline_output_has_viewbox(horizontal_line_image):
    config = VectorizerConfig(deskew_enabled=False)
    result = pipeline_run(horizontal_line_image, config)
    assert 'viewBox' in result.svg_string
    assert result.viewbox[2] > 0   # positive width
    assert result.viewbox[3] > 0   # positive height

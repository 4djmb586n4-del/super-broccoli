from __future__ import annotations

from typing import TypedDict


class LayerStyle(TypedDict):
    stroke: str
    stroke_width: float
    fill: str


LAYER_STYLES: dict[str, LayerStyle] = {
    'geometry':   {'stroke': '#000000', 'stroke_width': 0.5,  'fill': 'none'},
    'walls':      {'stroke': '#000000', 'stroke_width': 0.7,  'fill': 'none'},
    'dimensions': {'stroke': '#0000cc', 'stroke_width': 0.35, 'fill': 'none'},
    'text':       {'stroke': 'none',    'stroke_width': 0.0,  'fill': '#000000'},
    'hatching':   {'stroke': '#555555', 'stroke_width': 0.18, 'fill': 'none'},
    'symbols':    {'stroke': '#000000', 'stroke_width': 0.25, 'fill': 'none'},
}

DEFAULT_LAYER = 'geometry'


def style_for(layer: str) -> LayerStyle:
    return LAYER_STYLES.get(layer, LAYER_STYLES[DEFAULT_LAYER])

from __future__ import annotations

import numpy as np
import pytest

from vectorizer.config import VectorizerConfig
from vectorizer.m4_graph.build import build_graph


def _make_horizontal_skeleton(rows=50, cols=100, row=25):
    skel = np.zeros((rows, cols), dtype=bool)
    skel[row, 10:90] = True
    return skel


def test_graph_has_edges_for_line():
    skel = _make_horizontal_skeleton()
    G = build_graph(skel)
    assert G.number_of_edges() >= 1


def test_graph_edge_has_path_attribute():
    skel = _make_horizontal_skeleton()
    G = build_graph(skel)
    for u, v, data in G.edges(data=True):
        assert 'path' in data
        assert isinstance(data['path'], np.ndarray)
        assert data['path'].ndim == 2
        assert data['path'].shape[1] == 2
        break  # just check the first edge

from __future__ import annotations

import numpy as np
import networkx as nx

try:
    import skan
    _SKAN_AVAILABLE = True
except ImportError:
    _SKAN_AVAILABLE = False


def _build_with_skan(skeleton: np.ndarray) -> nx.Graph:
    """Use skan to extract branch table and convert to networkx."""
    sk = skan.Skeleton(skeleton, keep_images=False)
    branch_data = skan.summarize(sk, separator='-')

    G = nx.Graph()

    for _, row in branch_data.iterrows():
        src_idx = int(row['node-id-src'])
        dst_idx = int(row['node-id-dst'])

        # skan stores coordinates in the Skeleton.coordinates array
        src_coord = tuple(sk.coordinates[src_idx].astype(float))
        dst_coord = tuple(sk.coordinates[dst_idx].astype(float))

        # Retrieve the pixel path for this branch
        path_coords = sk.path_coordinates(int(row.name))

        G.add_node(src_coord)
        G.add_node(dst_coord)
        G.add_edge(
            src_coord,
            dst_coord,
            path=path_coords.astype(np.float32),
            length=float(row['branch-distance']),
        )

    return G


def _build_fallback(skeleton: np.ndarray) -> nx.Graph:
    """Minimal fallback when skan is unavailable: treat each connected run as one edge."""
    from scipy.ndimage import label as nd_label

    labeled, n = nd_label(skeleton)
    G = nx.Graph()
    for i in range(1, n + 1):
        yx = np.column_stack(np.where(labeled == i)).astype(np.float32)
        if len(yx) < 2:
            continue
        src = tuple(yx[0])
        dst = tuple(yx[-1])
        G.add_node(src)
        G.add_node(dst)
        G.add_edge(src, dst, path=yx, length=float(len(yx)))
    return G


def build_graph(skeleton: np.ndarray) -> nx.Graph:
    if _SKAN_AVAILABLE:
        try:
            return _build_with_skan(skeleton)
        except Exception:
            pass
    return _build_fallback(skeleton)

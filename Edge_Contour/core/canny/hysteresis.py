"""
Edge Tracking by Hysteresis.

Promotes weak pixels to strong (edge) if they are connected — directly or
through a chain of weak pixels — to at least one strong pixel.
All remaining weak pixels are suppressed to zero.
"""

import numpy as np
from collections import deque
from .threshold import STRONG_PIXEL, WEAK_PIXEL


def hysteresis(thresholded: np.ndarray) -> np.ndarray:
    """
    Finalise the edge map via hysteresis connectivity analysis.

    Parameters
    ----------
    thresholded : 2-D uint8 array produced by double_threshold()
                  (values: 0, WEAK_PIXEL=75, STRONG_PIXEL=255)

    Returns
    -------
    edges : 2-D uint8 array — final binary edge map (0 or 255)
    """
    rows, cols = thresholded.shape
    edges = np.zeros_like(thresholded, dtype=np.uint8)

    # Seed: all strong pixels are definite edges
    strong_rows, strong_cols = np.where(thresholded == STRONG_PIXEL)
    edges[strong_rows, strong_cols] = STRONG_PIXEL

    # BFS / flood-fill from every strong pixel
    queue = deque(zip(strong_rows.tolist(), strong_cols.tolist()))

    # 8-connectivity offsets
    neighbours = [(-1, -1), (-1, 0), (-1, 1),
                  ( 0, -1),          ( 0, 1),
                  ( 1, -1), ( 1, 0), ( 1, 1)]

    while queue:
        r, c = queue.popleft()
        for dr, dc in neighbours:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                if thresholded[nr, nc] == WEAK_PIXEL and edges[nr, nc] == 0:
                    edges[nr, nc] = STRONG_PIXEL          # promote weak → strong
                    queue.append((nr, nc))

    return edges
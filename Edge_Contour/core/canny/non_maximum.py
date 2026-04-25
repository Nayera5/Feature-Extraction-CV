"""
Non-Maximum Suppression (NMS).

For every pixel, keep its gradient magnitude only if it is the local
maximum along the gradient direction; otherwise suppress it to zero.
This thins the edge map to single-pixel-wide ridges.
"""

import numpy as np


def non_maximum_suppression(magnitude: np.ndarray,
                             direction: np.ndarray) -> np.ndarray:
    """
    Thin edges by suppressing non-maximum gradient pixels.

    Parameters
    ----------
    magnitude : 2-D float64 array — gradient magnitudes (0-255 scale)
    direction : 2-D float64 array — quantised angles (0, 45, 90, 135)

    Returns
    -------
    suppressed : 2-D float64 array, same shape, non-maxima zeroed out
    """
    rows, cols = magnitude.shape
    suppressed = np.zeros_like(magnitude, dtype=np.float64)

    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            angle = direction[i, j]
            mag   = magnitude[i, j]

            # Select the two neighbours along the gradient direction
            if angle == 0:          # horizontal edge  → compare left / right
                n1 = magnitude[i, j - 1]
                n2 = magnitude[i, j + 1]
            elif angle == 45:       # diagonal ↗↙
                n1 = magnitude[i - 1, j + 1]
                n2 = magnitude[i + 1, j - 1]
            elif angle == 90:       # vertical edge    → compare top / bottom
                n1 = magnitude[i - 1, j]
                n2 = magnitude[i + 1, j]
            else:                   # angle == 135, diagonal ↖↘
                n1 = magnitude[i - 1, j - 1]
                n2 = magnitude[i + 1, j + 1]

            if mag >= n1 and mag >= n2:
                suppressed[i, j] = mag
            # else: leave as 0 (already initialised)

    return suppressed
"""
area.py — Signed and unsigned area of a closed polygon (Shoelace / Gauss formula).

Works directly on the (x, y) contour points, so no chain code is needed.
"""

from __future__ import annotations

import numpy as np


def compute_area(contour: np.ndarray) -> float:
    """Return the area enclosed by *contour* using the Shoelace formula.

    Parameters
    ----------
    contour : ndarray, shape (N, 2)
        Ordered (x, y) coordinates.  The polygon is automatically closed
        (last point connected back to first).

    Returns
    -------
    float
        Unsigned area in square pixels.
    """
    if len(contour) < 3:
        return 0.0

    x = contour[:, 0].astype(float)
    y = contour[:, 1].astype(float)

    # Shoelace formula: A = 0.5 * |Σ (x_i * y_{i+1} - x_{i+1} * y_i)|
    area = 0.5 * abs(
        np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y)
    )
    return float(area)


def compute_signed_area(contour: np.ndarray) -> float:
    """Return the *signed* area.

    Positive → counter-clockwise winding.
    Negative → clockwise winding.
    """
    if len(contour) < 3:
        return 0.0

    x = contour[:, 0].astype(float)
    y = contour[:, 1].astype(float)

    return 0.5 * float(
        np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y)
    )
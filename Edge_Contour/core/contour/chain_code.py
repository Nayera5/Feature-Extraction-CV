"""
chain_code.py — Freeman 8-directional chain code from a contour.

The contour is expected as an (N, 2) integer array of (x, y) pixel coords.
"""

from __future__ import annotations

import numpy as np

# Freeman 8-connectivity direction mapping
# Direction index → (dx, dy)
#   3  2  1
#   4  *  0
#   5  6  7
_DIR_TO_DELTA: dict[int, tuple[int, int]] = {
    0: ( 1,  0),
    1: ( 1, -1),
    2: ( 0, -1),
    3: (-1, -1),
    4: (-1,  0),
    5: (-1,  1),
    6: ( 0,  1),
    7: ( 1,  1),
}

# (dx, dy) → direction index  (build inverse map)
_DELTA_TO_DIR: dict[tuple[int, int], int] = {v: k for k, v in _DIR_TO_DELTA.items()}


def compute_chain_code(contour: np.ndarray) -> list[int]:
    """Compute the Freeman 8-direction chain code of a closed contour.

    Parameters
    ----------
    contour : ndarray, shape (N, 2)
        Ordered (x, y) integer coordinates.  The contour is treated as
        closed (last point → first point is included).

    Returns
    -------
    list[int]
        Sequence of direction codes 0-7.  Length equals len(contour).
    """
    if len(contour) < 2:
        return []

    codes: list[int] = []
    n = len(contour)

    for i in range(n):
        x0, y0 = contour[i]
        x1, y1 = contour[(i + 1) % n]

        dx = int(np.sign(x1 - x0))
        dy = int(np.sign(y1 - y0))

        code = _DELTA_TO_DIR.get((dx, dy))
        if code is None:
            # Diagonal longer than 1 pixel — snap to closest 8-direction
            angle = np.arctan2(y1 - y0, x1 - x0)
            code = int(round(angle / (np.pi / 4))) % 8
        codes.append(code)

    return codes


def chain_code_to_string(codes: list[int]) -> str:
    """Return the chain code as a compact digit string, e.g. '01776543'."""
    return "".join(str(c) for c in codes)


def chain_code_difference(codes: list[int]) -> list[int]:
    """Return the first-difference (rotation-invariant) chain code.

    Each value is (codes[i] - codes[i-1]) mod 8.
    """
    if not codes:
        return []
    n = len(codes)
    return [(codes[i] - codes[i - 1]) % 8 for i in range(n)]
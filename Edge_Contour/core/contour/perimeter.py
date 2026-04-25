"""
perimeter.py — Perimeter estimation from a Freeman chain code.

For an 8-connected chain code:
  • Axis-aligned moves   (codes 0, 2, 4, 6) contribute 1 pixel distance.
  • Diagonal moves       (codes 1, 3, 5, 7) contribute √2 pixel distance.
"""

from __future__ import annotations

import math

_SQRT2 = math.sqrt(2.0)

# Even codes → straight; odd codes → diagonal
_STEP_LENGTH: dict[int, float] = {
    0: 1.0,
    1: _SQRT2,
    2: 1.0,
    3: _SQRT2,
    4: 1.0,
    5: _SQRT2,
    6: 1.0,
    7: _SQRT2,
}


def compute_perimeter(chain_code: list[int]) -> float:
    """Estimate perimeter (in pixels) from a Freeman 8-direction chain code.

    Parameters
    ----------
    chain_code : list[int]
        Sequence of direction codes 0-7.

    Returns
    -------
    float
        Perimeter length in pixels.
    """
    if not chain_code:
        return 0.0
    return sum(_STEP_LENGTH.get(c, 1.0) for c in chain_code)
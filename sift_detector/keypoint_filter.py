"""
keypoint_filter.py
------------------
Filters raw scale-space extrema by removing:

1. **Low-contrast responses** – DoG magnitude below a threshold.
2. **Edge responses** – keypoints located on edges rather than corners,
   detected using the ratio of principal curvatures (Harris-like criterion
   on the Hessian of the DoG image).

Reference: Lowe (2004) §4.1 and §4.2.
"""

import time
import numpy as np
from dataclasses import dataclass

from extrema_detection import RawKeypoint


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Keypoint:
    """Filtered, localised keypoint in the original image coordinate frame."""
    octave: int
    scale: int
    x: float          # column in original image coordinates
    y: float          # row    in original image coordinates
    size: float       # characteristic scale (σ) in pixels
    response: float   # DoG value


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def filter_keypoints(
    raw_keypoints: list[RawKeypoint],
    dog_pyramid: list[list[np.ndarray]],
    contrast_threshold: float = 0.04,
    edge_threshold: float = 10.0,
    sigma: float = 1.6,
    scales_per_octave: int = 3,
) -> tuple[list[Keypoint], dict]:
    """
    Remove low-contrast and edge-localised keypoints.

    Parameters
    ----------
    raw_keypoints      : Candidates from ``extrema_detection``.
    dog_pyramid        : DoG pyramid (same object used for extrema detection).
    contrast_threshold : Minimum |DoG| value after normalisation.
    edge_threshold     : Maximum principal-curvature ratio (r in Lowe 2004).
    sigma              : Base sigma used to build the Gaussian pyramid.
    scales_per_octave  : s parameter from pyramid construction.

    Returns
    -------
    keypoints : list[Keypoint]  – filtered and converted to image coordinates.
    info      : dict with timing, counts, and rejection breakdown.
    """
    t0 = time.perf_counter()

    low_contrast_rejected = 0
    edge_rejected = 0
    accepted: list[Keypoint] = []

    threshold = np.floor(0.5 * contrast_threshold / scales_per_octave * 255)

    for rk in raw_keypoints:
        dog_img = dog_pyramid[rk.octave][rk.scale]

        # ---- 1. Contrast filter -------------------------------------------
        pixel_val = dog_img[rk.row, rk.col]
        if abs(pixel_val) * 255 < threshold:
            low_contrast_rejected += 1
            continue

        # ---- 2. Edge filter (principal curvature ratio) -------------------
        if _is_on_edge(dog_img, rk.row, rk.col, edge_threshold):
            edge_rejected += 1
            continue

        # ---- Convert to original image coordinates ------------------------
        # Each octave halves resolution; we up-sampled by 2× at the start
        # so octave 0 is at 2× the input resolution.
        scale_factor = 2 ** (rk.octave - 1)   # maps back to input pixels
        x = rk.col * scale_factor
        y = rk.row * scale_factor

        # Characteristic scale in input-image pixels
        size = sigma * (2 ** (rk.scale / scales_per_octave)) * scale_factor * 2

        accepted.append(Keypoint(
            octave=rk.octave,
            scale=rk.scale,
            x=float(x),
            y=float(y),
            size=float(size),
            response=float(pixel_val),
        ))

    elapsed = time.perf_counter() - t0

    info = {
        "computation_time_sec": elapsed,
        "input_candidates": len(raw_keypoints),
        "low_contrast_rejected": low_contrast_rejected,
        "edge_rejected": edge_rejected,
        "accepted": len(accepted),
        "rejection_rate_pct": round(
            (1 - len(accepted) / max(len(raw_keypoints), 1)) * 100, 1
        ),
    }
    return accepted, info


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_on_edge(
    dog_img: np.ndarray,
    r: int,
    c: int,
    edge_threshold: float,
) -> bool:
    """
    Use the 2×2 Hessian at (r, c) to decide if the point is an edge response.

    The curvature ratio test:
        Tr(H)² / Det(H)  <  (r_threshold + 1)² / r_threshold
    """
    # Finite-difference Hessian
    dxx = (dog_img[r, c+1] - 2 * dog_img[r, c] + dog_img[r, c-1])
    dyy = (dog_img[r+1, c] - 2 * dog_img[r, c] + dog_img[r-1, c])
    dxy = ((dog_img[r+1, c+1] - dog_img[r+1, c-1] -
            dog_img[r-1, c+1] + dog_img[r-1, c-1]) / 4.0)

    trace = dxx + dyy
    det   = dxx * dyy - dxy ** 2

    if det <= 0:
        return True   # degenerate Hessian → treat as edge

    curvature_ratio = (trace ** 2) / det
    threshold_ratio = (edge_threshold + 1) ** 2 / edge_threshold
    return curvature_ratio >= threshold_ratio
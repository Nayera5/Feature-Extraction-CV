from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Tuple

import cv2
import numpy as np
from scipy.ndimage import convolve, gaussian_filter, maximum_filter


# ── Result container ──────────────────────────────────────────────────────

@dataclass
class HarrisResult:
    keypoints: List[cv2.KeyPoint]
    response_map: np.ndarray
    visualisation: np.ndarray
    computation_time_ms: float
    num_corners: int
    params: dict = field(default_factory=dict)


# ── Structure tensor ──────────────────────────────────────────────────────

def _compute_structure_tensor(
        image: np.ndarray,
        pre_sigma: float = 0.5,
        window_sigma: float = 1.5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns Sx2, Sy2, Sxy — the Gaussian-smoothed gradient products.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(float)

    # Light pre-smoothing to suppress noise before differentiation
    gray = gaussian_filter(gray, sigma=pre_sigma)

    sobel_x = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]], dtype=float)
    sobel_y = sobel_x.T

    Ix = convolve(gray, sobel_x)
    Iy = convolve(gray, sobel_y)

    # Smooth the gradient products with a Gaussian window
    Sx2 = gaussian_filter(Ix ** 2, sigma=window_sigma)
    Sy2 = gaussian_filter(Iy ** 2, sigma=window_sigma)
    Sxy = gaussian_filter(Ix * Iy, sigma=window_sigma)

    return Sx2, Sy2, Sxy


# ── NMS ───────────────────────────────────────────────────────────────────

def _nms(R: np.ndarray, min_dist: int) -> np.ndarray:
    """
    Dilation-based Non-Maximum Suppression.

    A pixel survives iff it equals the local maximum in a
    (2*min_dist+1) × (2*min_dist+1) neighbourhood AND is > 0.

    This works at every pixel including borders (maximum_filter uses
    reflect padding), so no corners are lost at image edges.
    """
    size = 2 * min_dist + 1
    local_max = maximum_filter(R, size=size, mode="reflect")
    # strict greater-than avoids keeping entire flat plateaus
    return (R == local_max) & (R > 0)


# ── Detection ─────────────────────────────────────────────────────────────

def _detect_corners(
        R: np.ndarray,
        threshold_rel: float,
        min_dist: int,
        max_corners: int,
) -> List[Tuple[int, int, float]]:
    """
    Returns a list of (x, y, strength) sorted by descending strength.

    threshold_rel : fraction of max(R_positive) used as cut-off.
                    e.g. 0.01 → keep pixels with R ≥ 1 % of the peak.
    """
    # Work only with positive Harris responses (negative = edge / flat)
    R_pos = np.where(R > 0, R, 0.0)

    if R_pos.max() == 0:
        return []

    threshold = threshold_rel * R_pos.max()
    R_thresh = np.where(R_pos >= threshold, R_pos, 0.0)

    # NMS
    mask = _nms(R_thresh, min_dist)
    ys, xs = np.where(mask)
    strengths = R_thresh[ys, xs]

    # Sort by strength, keep top N
    order = np.argsort(-strengths)[:max_corners]
    corners = [
        (int(xs[i]), int(ys[i]), float(strengths[i]))
        for i in order
    ]
    return corners


# ── Drawing ───────────────────────────────────────────────────────────────

def _draw(
        image: np.ndarray,
        corners: List[Tuple[int, int, float]],
) -> Tuple[np.ndarray, List[cv2.KeyPoint]]:
    vis = image.copy()
    keypoints = []
    for x, y, strength in corners:
        cv2.circle(vis, (x, y), 4, (0, 0, 255), -1, cv2.LINE_AA)
        keypoints.append(cv2.KeyPoint(float(x), float(y), 6.0))
    return vis, keypoints


# ── Public API ────────────────────────────────────────────────────────────

def detect_harris(
        image: np.ndarray,
        k: float = 0.04,
        threshold_rel: float = 0.01,
        gaussian_ksize: int = 5,      # kept for UI compatibility, not used internally
        gaussian_sigma: float = 1.5,  # maps to window_sigma
        min_dist: int = 5,
        max_corners: int = 2000,
) -> HarrisResult:
    """
    Run Harris corner detection on a BGR uint8 image.

    Parameters
    ----------
    k             : Harris sensitivity constant (0.04 – 0.06 typical).
    threshold_rel : Fraction of peak R used as threshold (e.g. 0.01 = 1 %).
    gaussian_sigma: Sigma for the structure-tensor Gaussian window.
    min_dist      : NMS neighbourhood radius in pixels.
    max_corners   : Maximum corners returned (sorted by strength).
    """
    start = time.perf_counter()

    Sx2, Sy2, Sxy = _compute_structure_tensor(
        image,
        pre_sigma=0.5,
        window_sigma=gaussian_sigma,
    )

    det_M   = Sx2 * Sy2 - Sxy ** 2
    trace_M = Sx2 + Sy2
    R       = det_M - k * (trace_M ** 2)

    corners = _detect_corners(R, threshold_rel, min_dist, max_corners)
    vis, keypoints = _draw(image, corners)

    elapsed_ms = (time.perf_counter() - start) * 1000.0

    return HarrisResult(
        keypoints=keypoints,
        response_map=R,
        visualisation=vis,
        computation_time_ms=elapsed_ms,
        num_corners=len(keypoints),
        params=dict(
            k=k,
            threshold_rel=threshold_rel,
            gaussian_sigma=gaussian_sigma,
            min_dist=min_dist,
            max_corners=max_corners,
        ),
    )
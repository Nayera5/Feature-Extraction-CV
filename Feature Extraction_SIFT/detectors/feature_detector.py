from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Tuple

import cv2
import numpy as np
from scipy.ndimage import convolve, gaussian_filter, maximum_filter


# ── Result container ──────────────────────────────────────────────────────

@dataclass
class FeatureResult:
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
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(float)
    gray = gaussian_filter(gray, sigma=pre_sigma)

    sobel_x = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]], dtype=float)
    sobel_y = sobel_x.T

    Ix = convolve(gray, sobel_x)
    Iy = convolve(gray, sobel_y)

    Sx2 = gaussian_filter(Ix ** 2, sigma=window_sigma)
    Sy2 = gaussian_filter(Iy ** 2, sigma=window_sigma)
    Sxy = gaussian_filter(Ix * Iy, sigma=window_sigma)

    return Sx2, Sy2, Sxy


# ── Eigenvalue & Harris response ─────────────────────────────────────────

def _compute_response(
        Sx2: np.ndarray,
        Sy2: np.ndarray,
        Sxy: np.ndarray,
        method: str = "harris",
        k: float = 0.04,
) -> np.ndarray:
    if method == "lambda":
        # λ-min (Shi-Tomasi)
        trace = Sx2 + Sy2
        diff  = Sx2 - Sy2
        discriminant = np.sqrt(np.maximum(diff ** 2 + 4.0 * Sxy ** 2, 0.0))
        return 0.5 * (trace - discriminant)
    else:
        # Harris
        det_M = Sx2 * Sy2 - Sxy ** 2
        trace_M = Sx2 + Sy2
        return det_M - k * (trace_M ** 2)


# ── NMS ───────────────────────────────────────────────────────────────────

def _nms(R: np.ndarray, min_dist: int) -> np.ndarray:
    size = 2 * min_dist + 1
    local_max = maximum_filter(R, size=size, mode="reflect")
    return (R == local_max) & (R > 0)


# ── Corner detection ──────────────────────────────────────────────────────

def _detect_corners(
        R: np.ndarray,
        threshold_rel: float,
        min_dist: int,
        max_corners: int,
) -> List[Tuple[int, int, float]]:
    R_pos = np.where(R > 0, R, 0.0)
    if R_pos.max() == 0:
        return []

    threshold = threshold_rel * R_pos.max()
    R_thresh = np.where(R_pos >= threshold, R_pos, 0.0)

    mask = _nms(R_thresh, min_dist)
    ys, xs = np.where(mask)
    strengths = R_thresh[ys, xs]

    order = np.argsort(-strengths)[:max_corners]
    return [(int(xs[i]), int(ys[i]), float(strengths[i])) for i in order]


# ── Drawing ───────────────────────────────────────────────────────────────

def _draw(image: np.ndarray, corners: List[Tuple[int, int, float]], method: str = "harris"):
    vis = image.copy()
    keypoints = []
    color = (0, 0, 255) if method == "harris" else (0, 255, 0)
    for x, y, _ in corners:
        cv2.circle(vis, (x, y), 3, color, -1, cv2.LINE_AA)
        keypoints.append(cv2.KeyPoint(float(x), float(y), 6.0))
    return vis, keypoints


# ── Public API ────────────────────────────────────────────────────────────

def detect_features(
        image: np.ndarray,
        method: str = "harris",
        k: float = 0.04,
        threshold_rel: float = 0.01,
        gaussian_sigma: float = 1.5,
        min_dist: int = 5,
        max_corners: int = 2000,
) -> FeatureResult:
    """
    Run Harris or λ- (Shi-Tomasi) corner detection.
    """
    start = time.perf_counter()

    Sx2, Sy2, Sxy = _compute_structure_tensor(image, pre_sigma=0.5, window_sigma=gaussian_sigma)
    R = _compute_response(Sx2, Sy2, Sxy, method=method, k=k)
    corners = _detect_corners(R, threshold_rel, min_dist, max_corners)
    vis, keypoints = _draw(image, corners, method=method)

    elapsed_ms = (time.perf_counter() - start) * 1000.0

    return FeatureResult(
        keypoints=keypoints,
        response_map=R,
        visualisation=vis,
        computation_time_ms=elapsed_ms,
        num_corners=len(keypoints),
        params=dict(
            method=method,
            k=k,
            threshold_rel=threshold_rel,
            gaussian_sigma=gaussian_sigma,
            min_dist=min_dist,
            max_corners=max_corners,
        ),
    )
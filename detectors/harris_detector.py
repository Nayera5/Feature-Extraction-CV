from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Tuple

import cv2
import numpy as np
from scipy.ndimage import convolve, gaussian_filter


@dataclass
class HarrisResult:
    keypoints: List[cv2.KeyPoint]
    response_map: np.ndarray
    visualisation: np.ndarray
    computation_time_ms: float
    num_corners: int
    params: dict = field(default_factory=dict)



def _compute_structure_tensor(image: np.ndarray):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    sobel_x = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]])
    sobel_y = np.array([[-1, -2, -1],
                        [0, 0, 0],
                        [1, 2, 1]])

    gray = gaussian_filter(gray.astype(float), sigma=0.5)

    Ix = convolve(gray, sobel_x)
    Iy = convolve(gray, sobel_y)

    Ix2 = Ix ** 2
    Iy2 = Iy ** 2
    Ixy = Ix * Iy

    sigma = 1.5
    Sx2 = gaussian_filter(Ix2, sigma=sigma)
    Sy2 = gaussian_filter(Iy2, sigma=sigma)
    Sxy = gaussian_filter(Ixy, sigma=sigma)

    return Sx2, Sy2, Sxy


def _detect_corners(R, threshold_rel, max_corners=2000):

    R = (R - R.min()) / (R.max() - R.min() + 1e-6)
    threshold = threshold_rel if threshold_rel else 0.05

    window_size = 6
    half = window_size // 2
    corners = []

    for y in range(half, R.shape[0] - half):
        for x in range(half, R.shape[1] - half):

            value = R[y, x]
            if value < threshold:
                continue

            local = R[y-half:y+half+1, x-half:x+half+1]

            #  STRICT MAX CONDITION
            if value == np.max(local):
                # Count how many times max appears
                if np.sum(local == value) == 1:
                    corners.append((x, y, value))

    # Sort and limit
    corners = sorted(corners, key=lambda x: x[2], reverse=True)
    corners = corners[:max_corners]

    return corners, R


def _draw(image, corners):
    vis = image.copy()

    keypoints = []

    for x, y, strength in corners:
        cv2.circle(vis, (x, y), 4, (0, 0, 255), -1)  # 🔴 RED
        keypoints.append(cv2.KeyPoint(float(x), float(y), 6))

    return vis, keypoints


# ---------------------------------------------------------------------------
# MAIN FUNCTION (used by controller)
# ---------------------------------------------------------------------------

def detect_harris(
        image: np.ndarray,
        k: float = 0.04,
        threshold_rel: float = 0.01,
        gaussian_ksize: int = 5,
        gaussian_sigma: float = 1.0,
        min_dist: int = 5,
        max_corners: int = 2000,
) -> HarrisResult:

    start = time.perf_counter()

    # Step 1: structure tensor
    Sx2, Sy2, Sxy = _compute_structure_tensor(image)

    # Step 2: Harris response
    det_M = Sx2 * Sy2 - Sxy ** 2
    trace_M = Sx2 + Sy2
    R = det_M - k * (trace_M ** 2)

    # Step 3: detect corners (your logic)
    corners, R_norm = _detect_corners(R, threshold_rel, max_corners)

    # Step 4: draw
    vis, keypoints = _draw(image, corners)

    end = time.perf_counter()

    return HarrisResult(
        keypoints=keypoints,
        response_map=R,
        visualisation=vis,
        computation_time_ms=(end - start) * 1000,
        num_corners=len(keypoints),
        params=dict(
            k=k,
            threshold_rel=threshold_rel,
            max_corners=max_corners
        )
    )
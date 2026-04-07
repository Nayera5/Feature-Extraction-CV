from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List

import cv2
import numpy as np
from scipy.ndimage import convolve, gaussian_filter


@dataclass
class LambdaResult:
    keypoints: List[cv2.KeyPoint]
    response_map: np.ndarray
    visualisation: np.ndarray
    computation_time_ms: float
    num_corners: int
    params: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Shared structure tensor
# ---------------------------------------------------------------------------

def _compute_structure_tensor(image):
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


# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------

def _detect_corners(R, threshold_rel, max_corners=2000):

    R = (R - R.min()) / (R.max() - R.min() + 1e-6)
    threshold = threshold_rel if threshold_rel else 0.1

    window_size = 6
    half = window_size // 2
    corners = []

    for y in range(half, R.shape[0] - half):
        for x in range(half, R.shape[1] - half):

            value = R[y, x]
            if value < threshold:
                continue

            local = R[y-half:y+half+1, x-half:x+half+1]

            if value == np.max(local):
                if np.sum(local == value) == 1:
                    corners.append((x, y, value))

    corners = sorted(corners, key=lambda x: x[2], reverse=True)
    corners = corners[:max_corners]

    return corners, R


def _draw(image, corners):
    vis = image.copy()
    keypoints = []

    for x, y, s in corners:
        cv2.circle(vis, (x, y), 4, (0, 255, 0), -1)  # 🟢 GREEN for lambda
        keypoints.append(cv2.KeyPoint(float(x), float(y), 6))

    return vis, keypoints


# ---------------------------------------------------------------------------
# MAIN FUNCTION
# ---------------------------------------------------------------------------

def detect_lambda(
        image: np.ndarray,
        threshold_rel: float = 0.1,
        gaussian_ksize: int = 5,
        gaussian_sigma: float = 1.0,
        min_dist: int = 5,
        max_corners: int = 2000,
):

    start = time.perf_counter()

    Sx2, Sy2, Sxy = _compute_structure_tensor(image)

    R = np.zeros_like(Sx2)

    # Shi-Tomasi: min eigenvalue
    for y in range(R.shape[0]):
        for x in range(R.shape[1]):
            M = np.array([[Sx2[y, x], Sxy[y, x]],
                          [Sxy[y, x], Sy2[y, x]]])
            eigvals = np.linalg.eigvals(M)
            R[y, x] = np.min(eigvals)

    corners, R_norm = _detect_corners(R, threshold_rel, max_corners)

    vis, keypoints = _draw(image, corners)

    end = time.perf_counter()

    return LambdaResult(
        keypoints=keypoints,
        response_map=R,
        visualisation=vis,
        computation_time_ms=(end - start) * 1000,
        num_corners=len(keypoints),
        params=dict(threshold_rel=threshold_rel)
    )
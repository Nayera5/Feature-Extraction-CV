"""
extrema_detection.py
--------------------
Detects scale-space extrema (local minima and maxima) in the DoG pyramid.

A pixel is an extremum if it is strictly greater than (or less than) all
26 neighbours: the 8 spatial neighbours in the same scale image **plus** the
9 neighbours in the scale above and the 9 neighbours in the scale below.

Reference: Lowe, D.G. (2004). "Distinctive Image Features from Scale-Invariant
           Keypoints." IJCV 60(2), 91–110.  §4.
"""

import time
import numpy as np
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class RawKeypoint:
    """Keypoint candidate before filtering."""
    octave: int
    scale: int          # index into the DoG octave (1 … n-2 are valid)
    row: int            # in the *octave* coordinate frame
    col: int            # in the *octave* coordinate frame
    value: float        # DoG response at this location


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_extrema(
    dog_pyramid: list[list[np.ndarray]],
    border: int = 5,
) -> tuple[list[RawKeypoint], dict]:
    """
    Find all scale-space extrema in the DoG pyramid.

    Parameters
    ----------
    dog_pyramid : Output of ``dog_pyramid.build_dog_pyramid``.
    border      : Pixels to ignore at the image boundary.

    Returns
    -------
    keypoints : list[RawKeypoint]
    info      : dict with timing and counts.
    """
    t0 = time.perf_counter()
    keypoints: list[RawKeypoint] = []

    for oct_idx, dog_octave in enumerate(dog_pyramid):
        # We need at least 3 consecutive DoG images to compare against
        if len(dog_octave) < 3:
            continue

        for scale_idx in range(1, len(dog_octave) - 1):
            prev_img  = dog_octave[scale_idx - 1]
            curr_img  = dog_octave[scale_idx]
            next_img  = dog_octave[scale_idx + 1]

            h, w = curr_img.shape

            # Stack the three images for fast 3D neighbourhood comparison
            cube = np.stack([prev_img, curr_img, next_img], axis=0)  # (3, H, W)

            rows = range(border, h - border)
            cols = range(border, w - border)

            for r in rows:
                for c in cols:
                    patch = cube[:, r-1:r+2, c-1:c+2]   # (3,3,3)
                    center = float(curr_img[r, c])

                    if _is_extremum(patch, center):
                        keypoints.append(RawKeypoint(
                            octave=oct_idx,
                            scale=scale_idx,
                            row=r,
                            col=c,
                            value=center,
                        ))

    elapsed = time.perf_counter() - t0
    info = {
        "computation_time_sec": elapsed,
        "total_candidates": len(keypoints),
        "per_octave_counts": _count_per_octave(keypoints, len(dog_pyramid)),
    }
    return keypoints, info


# ---------------------------------------------------------------------------
# Vectorised version (faster, used when border logic allows it)
# ---------------------------------------------------------------------------

def detect_extrema_fast(
    dog_pyramid: list[list[np.ndarray]],
    border: int = 5,
) -> tuple[list[RawKeypoint], dict]:
    """
    Vectorised extrema detection – same result as ``detect_extrema`` but
    uses numpy operations instead of Python loops for speed.
    """
    t0 = time.perf_counter()
    keypoints: list[RawKeypoint] = []

    for oct_idx, dog_octave in enumerate(dog_pyramid):
        if len(dog_octave) < 3:
            continue

        for scale_idx in range(1, len(dog_octave) - 1):
            prev_img = dog_octave[scale_idx - 1]
            curr_img = dog_octave[scale_idx]
            next_img = dog_octave[scale_idx + 1]

            h, w = curr_img.shape
            b = border

            # Valid region
            c = curr_img[b:h-b, b:w-b]

            # Build local maximum / minimum masks using image patches
            is_max = _local_max_mask(prev_img, curr_img, next_img, b, h, w)
            is_min = _local_min_mask(prev_img, curr_img, next_img, b, h, w)

            extrema_mask = is_max | is_min

            rows, cols = np.where(extrema_mask)
            for r, c_ in zip(rows, cols):
                actual_r = r + b
                actual_c = c_ + b
                keypoints.append(RawKeypoint(
                    octave=oct_idx,
                    scale=scale_idx,
                    row=int(actual_r),
                    col=int(actual_c),
                    value=float(curr_img[actual_r, actual_c]),
                ))

    elapsed = time.perf_counter() - t0
    info = {
        "computation_time_sec": elapsed,
        "total_candidates": len(keypoints),
        "per_octave_counts": _count_per_octave(keypoints, len(dog_pyramid)),
    }
    return keypoints, info


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _is_extremum(patch: np.ndarray, center: float) -> bool:
    """Check if center is strictly greater or smaller than all 26 neighbours."""
    flat = patch.flatten()
    flat_no_center = np.concatenate([flat[:13], flat[14:]])   # remove index 13 = centre
    return bool(center > flat_no_center.max()) or bool(center < flat_no_center.min())


def _local_max_mask(prev, curr, nxt, b, h, w):
    """Return boolean mask of local maxima in the valid region."""
    c = curr[b:h-b, b:w-b]
    # Compare against all 8 spatial neighbours in current scale
    mask = (
        (c > curr[b-1:h-b-1, b-1:w-b-1]) &
        (c > curr[b-1:h-b-1, b  :w-b  ]) &
        (c > curr[b-1:h-b-1, b+1:w-b+1]) &
        (c > curr[b  :h-b  , b-1:w-b-1]) &
        (c > curr[b  :h-b  , b+1:w-b+1]) &
        (c > curr[b+1:h-b+1, b-1:w-b-1]) &
        (c > curr[b+1:h-b+1, b  :w-b  ]) &
        (c > curr[b+1:h-b+1, b+1:w-b+1])
    )
    # Compare against 9 neighbours in prev scale
    for r_off in [-1, 0, 1]:
        for c_off in [-1, 0, 1]:
            mask &= (c > prev[b+r_off:h-b+r_off, b+c_off:w-b+c_off])
    # Compare against 9 neighbours in next scale
    for r_off in [-1, 0, 1]:
        for c_off in [-1, 0, 1]:
            mask &= (c > nxt[b+r_off:h-b+r_off, b+c_off:w-b+c_off])
    return mask


def _local_min_mask(prev, curr, nxt, b, h, w):
    """Return boolean mask of local minima in the valid region."""
    c = curr[b:h-b, b:w-b]
    mask = (
        (c < curr[b-1:h-b-1, b-1:w-b-1]) &
        (c < curr[b-1:h-b-1, b  :w-b  ]) &
        (c < curr[b-1:h-b-1, b+1:w-b+1]) &
        (c < curr[b  :h-b  , b-1:w-b-1]) &
        (c < curr[b  :h-b  , b+1:w-b+1]) &
        (c < curr[b+1:h-b+1, b-1:w-b-1]) &
        (c < curr[b+1:h-b+1, b  :w-b  ]) &
        (c < curr[b+1:h-b+1, b+1:w-b+1])
    )
    for r_off in [-1, 0, 1]:
        for c_off in [-1, 0, 1]:
            mask &= (c < prev[b+r_off:h-b+r_off, b+c_off:w-b+c_off])
    for r_off in [-1, 0, 1]:
        for c_off in [-1, 0, 1]:
            mask &= (c < nxt[b+r_off:h-b+r_off, b+c_off:w-b+c_off])
    return mask


def _count_per_octave(keypoints: list[RawKeypoint], num_octaves: int) -> list[int]:
    counts = [0] * num_octaves
    for kp in keypoints:
        if kp.octave < num_octaves:
            counts[kp.octave] += 1
    return counts
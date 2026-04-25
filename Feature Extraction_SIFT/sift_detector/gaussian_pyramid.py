"""
gaussian_pyramid.py
-------------------
Builds a Gaussian scale-space pyramid.

A Gaussian pyramid is a sequence of progressively blurred (and optionally
down-sampled) versions of an image.  Each *octave* contains `scales_per_octave`
images blurred with increasing σ values so that consecutive DoG images
span a constant ratio in scale space.
"""

import time
import numpy as np
import cv2


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_gaussian_pyramid(
    image: np.ndarray,
    num_octaves: int | None = None,
    scales_per_octave: int = 3,
    sigma: float = 1.6,
    assumed_blur: float = 0.5,
) -> tuple[list[list[np.ndarray]], dict]:
    """
    Build a multi-octave Gaussian pyramid.

    Parameters
    ----------
    image            : Input image (grayscale, uint8 or float32).
    num_octaves      : Number of octaves.  If None, computed from image size.
    scales_per_octave: Number of *extra* blur levels per octave (s in SIFT).
    sigma            : Target blur for the first image of each octave.
    assumed_blur     : Blur already present in the input image.

    Returns
    -------
    pyramid : list[list[np.ndarray]]
        pyramid[octave][scale] → blurred image (float32, [0,1]).
    info    : dict with timing and pyramid shape metadata.
    """
    t0 = time.perf_counter()

    # ---- pre-process -------------------------------------------------------
    img = _to_float(image)

    # Up-sample the base image by 2× (standard SIFT practice)
    base = cv2.resize(img, (img.shape[1] * 2, img.shape[0] * 2),
                      interpolation=cv2.INTER_LINEAR)

    # Remove assumed existing blur so the base has exactly σ_initial
    sigma_diff = max(np.sqrt(sigma ** 2 - (2 * assumed_blur) ** 2), 0.01)
    base = cv2.GaussianBlur(base, (0, 0), sigmaX=sigma_diff)

    # ---- determine pyramid shape -------------------------------------------
    if num_octaves is None:
        num_octaves = _auto_octaves(base.shape)

    # s intervals per octave → s+3 images per octave so DoG has s+2 images
    num_images_per_octave = scales_per_octave + 3
    k = 2 ** (1.0 / scales_per_octave)          # scale ratio between images
    sigmas = _compute_sigmas(sigma, num_images_per_octave, k)

    # ---- build pyramid ------------------------------------------------------
    pyramid: list[list[np.ndarray]] = []
    octave_img = base.copy()

    for oct_idx in range(num_octaves):
        octave_images: list[np.ndarray] = [octave_img]           # first image already at σ
        for scale_idx in range(1, num_images_per_octave):
            blurred = cv2.GaussianBlur(
                octave_images[scale_idx - 1],
                (0, 0),
                sigmaX=sigmas[scale_idx],
            )
            octave_images.append(blurred)
        pyramid.append(octave_images)

        # Next octave base = image at index `scales_per_octave` (half resolution)
        next_base = octave_images[scales_per_octave]
        octave_img = cv2.resize(
            next_base,
            (next_base.shape[1] // 2, next_base.shape[0] // 2),
            interpolation=cv2.INTER_NEAREST,
        )

    elapsed = time.perf_counter() - t0

    info = {
        "computation_time_sec": elapsed,
        "num_octaves": num_octaves,
        "num_images_per_octave": num_images_per_octave,
        "scales_per_octave": scales_per_octave,
        "base_sigma": sigma,
        "sigmas_within_octave": sigmas.tolist(),
        "octave_shapes": [
            [(img.shape[0], img.shape[1]) for img in oct]
            for oct in pyramid
        ],
    }
    return pyramid, info


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_float(image: np.ndarray) -> np.ndarray:
    """Convert to float32 in [0, 1]."""
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if image.dtype != np.float32:
        image = image.astype(np.float32)
        if image.max() > 1.0:
            image /= 255.0
    return image


def _auto_octaves(shape: tuple) -> int:
    """Compute number of octaves from image dimensions."""
    min_dim = min(shape[:2])
    return max(1, int(np.log2(min_dim)) - 1)


def _compute_sigmas(base_sigma: float, num_images: int, k: float) -> np.ndarray:
    """
    Compute the *incremental* σ values to apply between consecutive images.
    Each application blurs from the *previous* level, not from scratch.
    """
    total_sigmas = np.array([base_sigma * (k ** i) for i in range(num_images)])
    # Incremental blur needed: σ_inc² = σ_total[i]² − σ_total[i-1]²
    incremental = np.zeros(num_images)
    incremental[0] = 0.0   # first image is already at base_sigma
    for i in range(1, num_images):
        incremental[i] = np.sqrt(total_sigmas[i] ** 2 - total_sigmas[i - 1] ** 2)
    return incremental
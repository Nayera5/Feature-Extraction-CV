"""
dog_pyramid.py
--------------
Computes the Difference-of-Gaussians (DoG) pyramid from a Gaussian pyramid.

DoG is an efficient approximation of the Laplacian of Gaussian (LoG) and
serves as the scale-space representation used in SIFT for keypoint localisation.
"""

import time
import numpy as np


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_dog_pyramid(
    gaussian_pyramid: list[list[np.ndarray]],
) -> tuple[list[list[np.ndarray]], dict]:
    """
    Compute the DoG pyramid from a pre-built Gaussian pyramid.

    For each octave the DoG images are:
        DoG[i] = Gaussian[i+1] − Gaussian[i]

    Parameters
    ----------
    gaussian_pyramid : Output of ``gaussian_pyramid.build_gaussian_pyramid``.

    Returns
    -------
    dog_pyramid : list[list[np.ndarray]]
        dog_pyramid[octave][scale] → DoG image (float32).
        Each octave has (num_images_per_octave − 1) DoG images.
    info : dict with timing and shape metadata.
    """
    t0 = time.perf_counter()

    dog_pyramid: list[list[np.ndarray]] = []

    for oct_idx, octave_images in enumerate(gaussian_pyramid):
        dog_images: list[np.ndarray] = []
        for scale_idx in range(1, len(octave_images)):
            dog = octave_images[scale_idx].astype(np.float32) - \
                  octave_images[scale_idx - 1].astype(np.float32)
            dog_images.append(dog)
        dog_pyramid.append(dog_images)

    elapsed = time.perf_counter() - t0

    info = {
        "computation_time_sec": elapsed,
        "num_octaves": len(dog_pyramid),
        "dog_images_per_octave": [len(o) for o in dog_pyramid],
        "octave_shapes": [
            [(img.shape[0], img.shape[1]) for img in oct]
            for oct in dog_pyramid
        ],
        "value_ranges": [
            [(float(img.min()), float(img.max())) for img in oct]
            for oct in dog_pyramid
        ],
    }
    return dog_pyramid, info


# ---------------------------------------------------------------------------
# Visualisation helper (used by the UI)
# ---------------------------------------------------------------------------

def dog_to_display(dog_img: np.ndarray) -> np.ndarray:
    """
    Normalise a DoG image to uint8 [0,255] for display.
    Maps 0 → 128 (gray midpoint), positives → bright, negatives → dark.
    """
    shifted = dog_img + 0.5          # centre around 0.5
    clipped = np.clip(shifted, 0, 1)
    return (clipped * 255).astype(np.uint8)
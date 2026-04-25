"""
core.canny — full Canny edge detector implemented from scratch.

Convenience re-exports so callers can do:
    from core.canny import run_canny
"""

from .gaussian     import apply_gaussian_blur
from .gradients    import compute_gradients
from .non_maximum  import non_maximum_suppression
from .threshold    import double_threshold, STRONG_PIXEL, WEAK_PIXEL
from .hysteresis   import hysteresis


def run_canny(image,
              kernel_size: int   = 3,
              sigma:       float = 1.4,
              low_ratio:   float = 0.05,
              high_ratio:  float = 0.15):
    """
    Full Canny pipeline in one call.

    Parameters
    ----------
    image       : 2-D (grayscale) or 3-D (colour) uint8 numpy array
    kernel_size : Gaussian kernel size (odd integer, default 5)
    sigma       : Gaussian sigma (default 1.4)
    low_ratio   : double-threshold low  ratio  (default 0.05)
    high_ratio  : double-threshold high ratio  (default 0.15)

    Returns
    -------
    result : dict with keys
        'blurred'      — after Gaussian
        'magnitude'    — gradient magnitude
        'direction'    — quantised gradient direction
        'suppressed'   — after NMS
        'thresholded'  — after double threshold
        'edges'        — final binary edge map (uint8, 0 / 255)
        'low_thresh'   — absolute low  threshold
        'high_thresh'  — absolute high threshold
    """
    import numpy as np

    # Convert colour → grayscale if needed
    if image.ndim == 3:
        gray = (0.299  * image[:, :, 0] +
                0.587  * image[:, :, 1] +
                0.114  * image[:, :, 2]).astype(np.float64)
    else:
        gray = image.astype(np.float64)

    blurred     = apply_gaussian_blur(gray, kernel_size, sigma)
    mag, dirn, gx, gy = compute_gradients(blurred)
    suppressed  = non_maximum_suppression(mag, dirn)
    thresh, lo, hi = double_threshold(suppressed, low_ratio, high_ratio)
    edges       = hysteresis(thresh)

    return {
        "blurred":     blurred,
        "magnitude":   mag,
        "direction":   dirn,
        "gx":          gx,
        "gy":          gy,
        "suppressed":  suppressed,
        "thresholded": thresh,
        "edges":       edges,
        "low_thresh":  lo,
        "high_thresh": hi,
    }


__all__ = [
    "apply_gaussian_blur",
    "compute_gradients",
    "non_maximum_suppression",
    "double_threshold",
    "hysteresis",
    "run_canny",
    "STRONG_PIXEL",
    "WEAK_PIXEL",
]
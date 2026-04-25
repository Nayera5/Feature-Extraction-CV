"""
Double Thresholding.

Classifies every pixel in the suppressed gradient image as:
  • strong  (value == 255) — definitely an edge
  • weak    (value == 75 by default) — possible edge (decided by hysteresis)
  • zero    — not an edge
"""

import numpy as np

# Canonical pixel values used throughout the pipeline
STRONG_PIXEL: np.uint8 = np.uint8(255)
WEAK_PIXEL:   np.uint8 = np.uint8(75)


def double_threshold(suppressed: np.ndarray,
                     low_ratio:  float = 0.05,
                     high_ratio: float = 0.15):
    """
    Apply double thresholding to the NMS image.

    Parameters
    ----------
    suppressed : 2-D float64 array from non_maximum_suppression()
    low_ratio  : fraction of the image maximum used as the low threshold
    high_ratio : fraction of the image maximum used as the high threshold

    Returns
    -------
    thresholded : 2-D uint8 array with values 0, WEAK_PIXEL, or STRONG_PIXEL
    low_thresh  : absolute low  threshold value (float)
    high_thresh : absolute high threshold value (float)
    """
    if high_ratio <= low_ratio:
        raise ValueError("high_ratio must be greater than low_ratio.")

    img_max = suppressed.max()
    high_thresh = high_ratio * img_max
    low_thresh  = low_ratio  * img_max

    thresholded = np.zeros_like(suppressed, dtype=np.uint8)

    strong_mask = suppressed >= high_thresh
    weak_mask   = (suppressed >= low_thresh) & (~strong_mask)

    thresholded[strong_mask] = STRONG_PIXEL
    thresholded[weak_mask]   = WEAK_PIXEL

    return thresholded, low_thresh, high_thresh
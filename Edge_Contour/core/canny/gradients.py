"""
Gradient computation using Sobel operators.
Returns magnitude, direction (in degrees), and raw Gx/Gy components.
"""

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

# ── Sobel kernels ──────────────────────────────────────────────────────────────
SOBEL_X = np.array([[-1, 0, 1],
                     [-2, 0, 2],
                     [-1, 0, 1]], dtype=np.float64)

SOBEL_Y = np.array([[-1, -2, -1],
                     [ 0,  0,  0],
                     [ 1,  2,  1]], dtype=np.float64)


def convolve(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    k_h, k_w = kernel.shape
    pad_h, pad_w = k_h // 2, k_w // 2
    flipped = np.flipud(np.fliplr(kernel))   # flip for convolution

    # gray image
    if image.ndim == 2:
        padded = np.pad(image.astype(np.float64),
                        ((pad_h, pad_h), (pad_w, pad_w)), mode='constant')
        windows = sliding_window_view(padded, (k_h, k_w))
        return np.einsum('ijkl,kl->ij', windows, flipped)
    # colored image
    else:
        out = np.zeros(image.shape, dtype=np.float64)
        for c in range(image.shape[2]):
            padded = np.pad(image[:, :, c].astype(np.float64),
                            ((pad_h, pad_h), (pad_w, pad_w)), mode='edge')
            windows = sliding_window_view(padded, (k_h, k_w))
            out[:, :, c] = np.einsum('ijkl,kl->ij', windows, flipped)
        return np.clip(out, 0, 255).astype(np.uint8)  # clipping is better for colored images


def compute_gradients(blurred: np.ndarray):
    """
    Compute image gradients using Sobel filters.

    Parameters
    ----------
    blurred : 2-D float64 array (output of Gaussian blur)

    Returns
    -------
    magnitude  : gradient magnitude  (float64, same shape)
    direction  : gradient angle in degrees, quantised to 0/45/90/135
    gx         : raw horizontal gradient
    gy         : raw vertical gradient
    """
    gx = convolve(blurred, SOBEL_X)
    gy = convolve(blurred, SOBEL_Y)

    magnitude = np.hypot(gx, gy)
    magnitude = (magnitude / magnitude.max() * 255.0) if magnitude.max() > 0 else magnitude

    # Angle in degrees [0, 180)
    angle_rad = np.arctan2(gy, gx)
    angle_deg = np.degrees(angle_rad) % 180

    # Quantise to 4 directions: 0, 45, 90, 135
    direction = np.zeros_like(angle_deg, dtype=np.float64)
    direction[((angle_deg >= 0)   & (angle_deg < 22.5))  |
              ((angle_deg >= 157.5) & (angle_deg < 180))] = 0
    direction[(angle_deg >= 22.5)  & (angle_deg < 67.5)]  = 45
    direction[(angle_deg >= 67.5)  & (angle_deg < 112.5)] = 90
    direction[(angle_deg >= 112.5) & (angle_deg < 157.5)] = 135

    return magnitude, direction, gx, gy
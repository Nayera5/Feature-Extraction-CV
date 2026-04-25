"""
Gaussian Blur — noise reduction step for Canny edge detection.
Builds and applies a Gaussian kernel entirely with NumPy (no OpenCV blur).
"""

import numpy as np


def _gaussian_kernel(size: int, sigma: float) -> np.ndarray:
    """
    Generate a 2-D Gaussian kernel.

    Parameters
    ----------
    size  : kernel side length (must be odd)
    sigma : standard deviation of the Gaussian

    Returns
    -------
    kernel : (size × size) float64 array, sum == 1
    """
    if size % 2 == 0:
        raise ValueError("Kernel size must be odd.")

    ax = np.arange(-(size // 2), size // 2 + 1, dtype=np.float64)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx ** 2 + yy ** 2) / (2.0 * sigma ** 2))
    return kernel / kernel.sum()


def _convolve2d(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Manual 2-D convolution with zero-padding (no scipy / cv2).

    Parameters
    ----------
    image  : 2-D float64 array
    kernel : 2-D float64 array (assumed square, odd size)

    Returns
    -------
    output : same shape as *image*
    """
    kh, kw = kernel.shape
    ph, pw = kh // 2, kw // 2

    padded = np.pad(image, ((ph, ph), (pw, pw)), mode="reflect")
    output = np.zeros_like(image, dtype=np.float64)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            output[i, j] = (padded[i:i + kh, j:j + kw] * kernel).sum()

    return output


def apply_gaussian_blur(image: np.ndarray,
                        kernel_size: int = 5,
                        sigma: float = 1.4) -> np.ndarray:
    """
    Apply Gaussian blur to a grayscale image.

    Parameters
    ----------
    image       : 2-D uint8 or float array
    kernel_size : side length of the Gaussian kernel (default 5)
    sigma       : standard deviation (default 1.4)

    Returns
    -------
    blurred : 2-D float64 array, same shape as *image*
    """
    img = image.astype(np.float64)
    kernel = _gaussian_kernel(kernel_size, sigma)
    return _convolve2d(img, kernel)
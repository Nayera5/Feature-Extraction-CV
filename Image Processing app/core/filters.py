import numpy as np
from core.image_manager import ImageManager


def apply_filter(image, filter_type):
    if filter_type == "Average (3x3)":
        return average_filter(image)

    elif filter_type == "Gaussian (3x3)":
        return gaussian_filter(image)

    elif filter_type == "Median (3x3)":
        return median_filter(image)

    return image


def average_filter(image):
    kernel = np.ones((3, 3)) / 9
    return ImageManager.convolve(image, kernel)


def gaussian_filter(image):
    kernel = np.array([
        [1, 2, 1],
        [2, 4, 2],
        [1, 2, 1]
    ]) / 16
    return ImageManager.convolve(image, kernel)


def median_filter(image):
    from numpy.lib.stride_tricks import sliding_window_view
    padded = np.pad(image, ((1, 1), (1, 1), (0, 0)), mode='edge')
    # windows shape: (H, W, C, 3, 3)
    windows = sliding_window_view(padded, (3, 3), axis=(0, 1))
    # median over the two kernel axes → shape (H, W, C)
    return np.median(windows, axis=(-2, -1)).astype(np.uint8)
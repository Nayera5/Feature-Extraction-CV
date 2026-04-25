import numpy as np
import cv2
from core.frequency import _gaussian_mask, _fft, _ifft


def _resize_to_match(img1: np.ndarray, img2: np.ndarray) -> tuple:
    """Resize img2 to match img1's spatial dimensions."""
    h, w = img1.shape[:2]
    img2_resized = cv2.resize(img2, (w, h), interpolation=cv2.INTER_AREA)
    return img1, img2_resized


def _apply_gaussian_filter_freq(image: np.ndarray, cutoff: int, low_pass: bool) -> np.ndarray:
    """
    Apply a Gaussian low-pass or high-pass filter to an image.
    Returns uint8 result.
    """
    is_color = len(image.shape) == 3
    channels = cv2.split(image) if is_color else [image]
    shape = channels[0].shape
    mask = _gaussian_mask(shape, cutoff, low_pass)

    filtered = []
    for ch in channels:
        ch_float = ch.astype(np.float32) / 255.0
        f_shift = _fft(ch_float)
        f_filtered = f_shift * mask
        result = _ifft(f_filtered)
        filtered.append(result)

    return cv2.merge(filtered) if is_color else filtered[0]


def create_hybrid_image(
    image1: np.ndarray,
    image2: np.ndarray,
    low_cutoff: int = 30,
    high_cutoff: int = 20,
    alpha: float = 0.5,
    low_pass1: bool = True,   # True = LPF on image1, False = HPF on image1
    low_pass2: bool = False,  # True = LPF on image2, False = HPF on image2
) -> tuple:
    """
    Create a hybrid image combining filtered image1 and filtered image2.

    Parameters
    ----------
    image1     : First image.
    image2     : Second image.
    low_cutoff : Cutoff radius for image1's filter.
    high_cutoff: Cutoff radius for image2's filter.
    alpha      : Blending weight. Output = alpha * img1_filtered + (1-alpha) * img2_filtered.
    low_pass1  : If True apply LPF to image1, else HPF.
    low_pass2  : If True apply LPF to image2, else HPF.

    Returns
    -------
    hybrid         : uint8 hybrid image.
    img1_filtered  : uint8 filtered image1.
    img2_filtered  : uint8 filtered image2.
    """
    if image1 is None or image2 is None:
        raise ValueError("Both images must be provided.")

    # Match sizes and colour spaces
    image1, image2 = _resize_to_match(image1, image2)
    is_color1 = len(image1.shape) == 3
    is_color2 = len(image2.shape) == 3

    if is_color1 and not is_color2:
        image2 = cv2.cvtColor(image2, cv2.COLOR_GRAY2BGR)
    elif is_color2 and not is_color1:
        image1 = cv2.cvtColor(image1, cv2.COLOR_GRAY2BGR)

    # Filter each image with its own settings
    img1_filtered = _apply_gaussian_filter_freq(image1, cutoff=low_cutoff,  low_pass=low_pass1)
    img2_filtered = _apply_gaussian_filter_freq(image2, cutoff=high_cutoff, low_pass=low_pass2)

    # Blend
    low_f  = img1_filtered.astype(np.float32)
    high_f = img2_filtered.astype(np.float32)
    hybrid_f = alpha * low_f + (1.0 - alpha) * high_f
    hybrid = np.clip(hybrid_f, 0, 255).astype(np.uint8)

    return hybrid, img1_filtered, img2_filtered


def visualize_hybrid_scales(hybrid: np.ndarray, scales: int = 5) -> list:
    """
    Return progressively downscaled versions of the hybrid image
    to simulate the distance effect.
    """
    results = [hybrid]
    current = hybrid.copy()
    for _ in range(scales - 1):
        h, w = current.shape[:2]
        if h < 20 or w < 20:
            break
        current = cv2.resize(current, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
        results.append(current)
    return results
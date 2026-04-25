import numpy as np
import cv2

def _to_gray_float(image: np.ndarray) -> np.ndarray:
    """Convert image to grayscale float32 in [0,1]."""
    if image is None:
        raise ValueError("Input image is None.")
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    return gray.astype(np.float32) / 255.0


def _fft(image_float: np.ndarray):
    """Return shifted DFT of a float image."""
    f = np.fft.fft2(image_float)
    return np.fft.fftshift(f)


def _ifft(f_shifted: np.ndarray) -> np.ndarray:
    """Return spatial image from shifted DFT, clipped to [0,255] uint8."""
    f = np.fft.ifftshift(f_shifted)
    result = np.fft.ifft2(f)
    result = np.abs(result)
    result = np.clip(result * 255.0, 0, 255).astype(np.uint8)
    return result


def _ideal_circle_mask(shape: tuple, cutoff: int, low_pass: bool) -> np.ndarray:
    """Create an ideal circular low-pass or high-pass mask."""
    rows, cols = shape
    crow, ccol = rows // 2, cols // 2
    y, x = np.ogrid[:rows, :cols]
    dist = np.sqrt((x - ccol) ** 2 + (y - crow) ** 2)
    mask = (dist <= cutoff).astype(np.float32)
    return mask if low_pass else 1.0 - mask


def _gaussian_mask(shape: tuple, cutoff: int, low_pass: bool) -> np.ndarray:
    """Create a Gaussian low-pass or high-pass mask."""
    rows, cols = shape
    crow, ccol = rows // 2, cols // 2
    y, x = np.ogrid[:rows, :cols]
    dist_sq = (x - ccol) ** 2 + (y - crow) ** 2
    mask = np.exp(-dist_sq / (2 * cutoff ** 2)).astype(np.float32)
    return mask if low_pass else 1.0 - mask


def _butterworth_mask(shape: tuple, cutoff: int, order: int, low_pass: bool) -> np.ndarray:
    """Create a Butterworth low-pass or high-pass mask."""
    rows, cols = shape
    crow, ccol = rows // 2, cols // 2
    y, x = np.ogrid[:rows, :cols]
    dist = np.sqrt((x - ccol) ** 2 + (y - crow) ** 2)
    # Avoid division by zero
    dist[crow, ccol] = 1e-6
    if low_pass:
        mask = 1.0 / (1.0 + (dist / cutoff) ** (2 * order))
    else:
        mask = 1.0 / (1.0 + (cutoff / dist) ** (2 * order))
    return mask.astype(np.float32)


def apply_frequency_filter(
    image: np.ndarray,
    filter_type: str = "ideal",   # "ideal" | "gaussian" | "butterworth"
    pass_type: str = "low",       # "low" | "high"
    cutoff: int = 30,
    order: int = 2,               # only for butterworth
) -> tuple:
    """
    Apply a frequency-domain filter to a grayscale or colour image.

    Parameters
    ----------
    image      : Input BGR or grayscale image (uint8).
    filter_type: "ideal", "gaussian", or "butterworth".
    pass_type  : "low" or "high".
    cutoff     : Cut-off frequency (radius in pixels).
    order      : Butterworth filter order (ignored for other types).

    Returns
    -------
    filtered_image : uint8 result image (same channels as input).
    magnitude_spectrum : uint8 log-magnitude image of the DFT (for display).
    """
    is_color = len(image.shape) == 3
    low_pass = pass_type == "low"

    # ---- build channels to process ----
    if is_color:
        channels = cv2.split(image)
    else:
        channels = [image]

    shape = channels[0].shape

    # ---- build mask ----
    if filter_type == "ideal":
        mask = _ideal_circle_mask(shape, cutoff, low_pass)
    elif filter_type == "gaussian":
        mask = _gaussian_mask(shape, cutoff, low_pass)
    elif filter_type == "butterworth":
        mask = _butterworth_mask(shape, cutoff, order, low_pass)
    else:
        raise ValueError(f"Unknown filter_type: {filter_type!r}")

    # ---- process each channel ----
    filtered_channels = []
    magnitude_spectra = []

    for ch in channels:
        ch_float = ch.astype(np.float32) / 255.0
        f_shift = _fft(ch_float)

        # magnitude spectrum for visualisation (use first channel only)
        mag = np.log1p(np.abs(f_shift))
        mag = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        magnitude_spectra.append(mag)

        # apply mask
        f_filtered = f_shift * mask
        result_ch = _ifft(f_filtered)
        filtered_channels.append(result_ch)

    # ---- merge / return ----
    if is_color:
        filtered_image = cv2.merge(filtered_channels)
        magnitude_spectrum = cv2.merge(magnitude_spectra)
    else:
        filtered_image = filtered_channels[0]
        magnitude_spectrum = magnitude_spectra[0]

    return filtered_image, magnitude_spectrum


def get_magnitude_spectrum(image: np.ndarray) -> np.ndarray:
    """
    Utility: return the log-magnitude DFT spectrum of an image (for display).
    Works on grayscale or BGR images.
    """
    gray_float = _to_gray_float(image)
    f_shift = _fft(gray_float)
    mag = np.log1p(np.abs(f_shift))
    mag = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return mag
"""
image_manager.py — shared image I/O and conversion utilities.

Used by all controllers so there is a single place to change
how images are loaded, saved, and converted to Qt objects.
"""

from __future__ import annotations

import numpy as np
from pathlib import Path

from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt


# ── Loading ────────────────────────────────────────────────────────────────────

def load_image(path: str | Path) -> np.ndarray:
    """
    Load an image from *path* and return it as a uint8 NumPy array.

    Colour images  → shape (H, W, 3)  RGB
    Grayscale      → shape (H, W)

    Uses only Pillow so we have no hidden OpenCV dependency.
    """
    from PIL import Image

    img = Image.open(str(path))

    if img.mode == "RGBA":
        img = img.convert("RGB")

    arr = np.array(img, dtype=np.uint8)
    return arr


def save_image(array: np.ndarray, path: str | Path) -> None:
    """Save a NumPy array to *path* (format inferred from extension)."""
    from PIL import Image

    if array.dtype != np.uint8:
        # Normalise to 0-255 before saving
        mn, mx = array.min(), array.max()
        if mx > mn:
            array = ((array - mn) / (mx - mn) * 255).astype(np.uint8)
        else:
            array = np.zeros_like(array, dtype=np.uint8)

    Image.fromarray(array).save(str(path))


# ── Grayscale conversion ───────────────────────────────────────────────────────

def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert a colour (H, W, 3) array to (H, W) grayscale uint8."""
    if image.ndim == 2:
        return image
    return (0.299  * image[:, :, 0] +
            0.587  * image[:, :, 1] +
            0.114  * image[:, :, 2]).astype(np.uint8)


# ── NumPy ↔ Qt conversions ─────────────────────────────────────────────────────

def _normalise_u8(arr: np.ndarray) -> np.ndarray:
    """Ensure array is uint8 in [0, 255], normalising if needed."""
    if arr.dtype == np.uint8:
        return arr
    mn, mx = arr.min(), arr.max()
    if mx > mn:
        return ((arr - mn) / (mx - mn) * 255).astype(np.uint8)
    return np.zeros_like(arr, dtype=np.uint8)


def array_to_qimage(arr: np.ndarray) -> QImage:
    """
    Convert a NumPy array to QImage.
    Supports:
      • (H, W)      → Grayscale8
      • (H, W, 3)   → RGB888
    """
    arr = _normalise_u8(arr)

    if arr.ndim == 2:
        h, w = arr.shape
        # QImage requires the data to be contiguous
        data = np.ascontiguousarray(arr)
        return QImage(data.data, w, h, w, QImage.Format_Grayscale8)

    elif arr.ndim == 3 and arr.shape[2] == 3:
        h, w, _ = arr.shape
        data = np.ascontiguousarray(arr)
        return QImage(data.data, w, h, 3 * w, QImage.Format_RGB888)

    else:
        raise ValueError(f"Unsupported array shape: {arr.shape}")


def to_qpixmap(arr: np.ndarray, max_size: int | None = None) -> QPixmap:
    """
    Convert a NumPy array to QPixmap, optionally scaled to *max_size*.

    Parameters
    ----------
    arr      : 2-D or 3-D uint8 / float array
    max_size : if given, scale the pixmap so neither dimension exceeds this
    """
    qimg = array_to_qimage(arr)
    pix  = QPixmap.fromImage(qimg)

    if max_size is not None and (pix.width() > max_size or pix.height() > max_size):
        pix = pix.scaled(max_size, max_size,
                         Qt.KeepAspectRatio,
                         Qt.SmoothTransformation)
    return pix


def qpixmap_to_array(pixmap: QPixmap) -> np.ndarray:
    """Convert a QPixmap back to a (H, W, 4) RGBA uint8 NumPy array."""
    qimg = pixmap.toImage().convertToFormat(QImage.Format_RGBA8888)
    w, h = qimg.width(), qimg.height()
    ptr  = qimg.bits()
    ptr.setsize(h * w * 4)
    return np.frombuffer(ptr, dtype=np.uint8).reshape((h, w, 4))
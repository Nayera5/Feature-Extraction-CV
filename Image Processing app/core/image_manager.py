import cv2
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


class ImageManager:
    def __init__(self):
        self.original_image = None
        self.current_image = None
        self.gray_image = None

    @staticmethod
    def convolve(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        k_h, k_w = kernel.shape
        pad_h, pad_w = k_h // 2, k_w // 2
        flipped = np.flipud(np.fliplr(kernel))

        if image.ndim == 2:
            padded = np.pad(image.astype(np.float64),
                            ((pad_h, pad_h), (pad_w, pad_w)), mode='constant')
            windows = sliding_window_view(padded, (k_h, k_w))
            return np.einsum('ijkl,kl->ij', windows, flipped)
        else:
            out = np.zeros(image.shape, dtype=np.float64)
            for c in range(image.shape[2]):
                padded = np.pad(image[:, :, c].astype(np.float64),
                                ((pad_h, pad_h), (pad_w, pad_w)), mode='edge')
                windows = sliding_window_view(padded, (k_h, k_w))
                out[:, :, c] = np.einsum('ijkl,kl->ij', windows, flipped)
            return np.clip(out, 0, 255).astype(np.uint8)

    def read_image(self, path):
        image = cv2.imread(path)
        if image is None:
            raise FileNotFoundError(f"Image not found at path: {path}")
        self.original_image = image
        self.current_image = image.copy()
        self.gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # always compute once on load
        return self.current_image

    def reset_image(self):
        if self.original_image is not None:
            self.current_image = self.original_image.copy()
            self.gray_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
            return self.current_image
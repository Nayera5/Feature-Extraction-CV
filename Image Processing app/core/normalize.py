import numpy as np
import cv2


def normalize_image(image: np.ndarray) -> np.ndarray:
   
    image = image.astype(np.float64)

    if image.ndim == 2:
        # Grayscale
        normalized = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
    else:
        # Colour: normalize each channel independently
        normalized = np.zeros_like(image)
        for c in range(image.shape[2]):
            normalized[:, :, c] = cv2.normalize(image[:, :, c], None, 0, 255, cv2.NORM_MINMAX)

    return normalized.astype(np.uint8)

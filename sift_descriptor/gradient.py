import numpy as np


# -----------------------------
# Keypoint container
# -----------------------------
class DescriptorKeypoint:
    def __init__(self, x, y, scale, orientation, descriptor):
        self.x = float(x)
        self.y = float(y)
        self.scale = float(scale)
        self.orientation = float(orientation)
        self.descriptor = descriptor.astype(np.float32)


# -----------------------------
# 1. Gradient (Central Difference)
# -----------------------------
def compute_gradients(img):
    gx = np.zeros_like(img, dtype=np.float32)
    gy = np.zeros_like(img, dtype=np.float32)

    gx[:, 1:-1] = img[:, 2:] - img[:, :-2]
    gy[1:-1, :] = img[2:, :] - img[:-2, :]

    return gx, gy

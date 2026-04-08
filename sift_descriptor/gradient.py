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
    gy, gx = np.gradient(img.astype(np.float32))
    return gx, gy
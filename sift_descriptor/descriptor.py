import time
import numpy as np

from .gradient import DescriptorKeypoint, compute_gradients
from .orientation import dominant_orientation

def extract_patch(ix, iy, x, y, orientation, size=16):   # extract 16x16 patch aligned to dominant orientation لو الصورة اتلفت descriptor يفضل ثابت
    mag = np.zeros((size, size))
    ang = np.zeros((size, size))

    theta = np.deg2rad(orientation)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    half = size // 2

    for r in range(size):
        for c in range(size):

            u = c - half
            v = r - half

            xr = int(x + (u * cos_t - v * sin_t))
            yr = int(y + (u * sin_t + v * cos_t))

            if 1 <= xr < ix.shape[1]-1 and 1 <= yr < ix.shape[0]-1:
                gx, gy = ix[yr, xr], iy[yr, xr]

                mag[r, c] = np.hypot(gx, gy)
                ang[r, c] = (np.degrees(np.arctan2(gy, gx)) - orientation) % 360

    return mag, ang


def build_descriptor(mag, ang):
    desc = []

    for i in range(0, 16, 4):
        for j in range(0, 16, 4):

            hist = np.zeros(8)

            for r in range(4):
                for c in range(4):
                    m = mag[i+r, j+c]
                    a = ang[i+r, j+c]

                    bin_idx = int(a / 45) % 8
                    hist[bin_idx] += m

            desc.extend(hist)

    return np.array(desc)


def normalize(desc):
    # if np.linalg.norm(desc) != 0:
    #     desc = desc / np.linalg.norm(desc)

    # desc = np.clip(desc, 0, 0.2)

    if np.linalg.norm(desc) != 0:
        desc = desc / np.linalg.norm(desc)

    return desc

# def normalize(desc):
#     norm = np.linalg.norm(desc)
#     if norm > 0:
#         desc = desc / norm
#     desc = np.clip(desc, 0, 0.2)      # suppress large values
#     norm2 = np.linalg.norm(desc)      # renormalize after clipping
#     if norm2 > 0:
#         desc = desc / norm2
#     return desc


def generate_descriptors(keypoints, gaussian_pyramid):
    start = time.time()
    results = []

    for kp in keypoints:
        octave = int(kp.octave)
        
        if octave >= len(gaussian_pyramid):
            continue

        layer = gaussian_pyramid[octave][int(kp.scale) + 1]
        ix, iy = compute_gradients(layer)

        scale_factor = 2 ** (octave - 1)
        x = int(kp.x / scale_factor)
        y = int(kp.y / scale_factor)

        # x = int(kp.x)
        # y = int(kp.y)
        print("fffffffffffffffffffffffffffff")
        if x < 8 or y < 8 or x >= layer.shape[1]-8 or y >= layer.shape[0]-8:
            # Optional: try to extract with padding instead of skipping
            # Or use a smaller radius for edge keypoints
            continue
        print("uuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuu")
        orientation = dominant_orientation(ix, iy, x, y, radius=8)

        mag, ang = extract_patch(ix, iy, x, y, orientation)

        desc = normalize(build_descriptor(mag, ang))

        results.append(DescriptorKeypoint(kp.x, kp.y, kp.size, orientation, desc))

    end = time.time()

    return results, {
        "time": end - start,
        "input": len(keypoints),
        "output": len(results)
    }

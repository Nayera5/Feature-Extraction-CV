import numpy as np

# -----------------------------
# 2. Orientation Assignment
# -----------------------------
def dominant_orientation(ix, iy, x, y, radius, bins=36):
    hist = np.zeros(bins)

    for yy in range(y - radius, y + radius + 1):
        for xx in range(x - radius, x + radius + 1):

            gx, gy = ix[yy, xx], iy[yy, xx]
            mag = np.hypot(gx, gy)
            angle = (np.degrees(np.arctan2(gy, gx)) + 360) % 360  

            bin_idx = int(angle / 360 * bins)  # make 36 bins
            hist[bin_idx] += mag

    return np.argmax(hist) * (360 / bins)

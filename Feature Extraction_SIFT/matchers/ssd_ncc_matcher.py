import numpy as np
import time

def match_ssd(descriptors_A, descriptors_B, ratio_thresh=0.75):
    matches_A2B = []
    matches_B2A = []

    # A → B
    for i, f in enumerate(descriptors_A):
        diff = descriptors_B - f
        ssd = np.sum(diff ** 2, axis=1)

        if len(ssd) < 2:
            continue

        idx = np.argsort(ssd)
        best, second = ssd[idx[0]], ssd[idx[1]]

        if second > 0 and (best / second) < ratio_thresh:
            matches_A2B.append((i, int(idx[0])))

    # B → A
    for j, f in enumerate(descriptors_B):
        diff = descriptors_A - f
        ssd = np.sum(diff ** 2, axis=1)

        if len(ssd) < 2:
            continue

        idx = np.argsort(ssd)
        best, second = ssd[idx[0]], ssd[idx[1]]

        if second > 0 and (best / second) < ratio_thresh:
            matches_B2A.append((int(idx[0]), j))

    # Cross-check
    matches_B2A_set = set(matches_B2A)

    matches = []
    for m in matches_A2B:
        if m in matches_B2A_set:
            matches.append(m)

    print(f"Cross-check matches: {len(matches)}")
    return matches




def match_ncc(descriptors_A, descriptors_B, ratio_thresh=0.9):
    matches = []
    start = time.time()

    if len(descriptors_A) == 0 or len(descriptors_B) < 2:
        return matches

    # Pre-normalize all B descriptors once (faster than per-pair)
    B_mean = descriptors_B.mean(axis=1, keepdims=True)
    B_std  = descriptors_B.std(axis=1, keepdims=True) + 1e-8
    B_norm = (descriptors_B - B_mean) / B_std        # shape: (N_B, D)

    for i, f in enumerate(descriptors_A):
        if len(B_norm) < 2:
            continue

        f_std = np.std(f) + 1e-8
        f_norm = (f - np.mean(f)) / f_std            # shape: (D,)

        # Cosine-like normalized cross-correlation in [-1, 1]
        ncc_scores = (B_norm @ f_norm) / float(f_norm.size)  # shape: (N_B,)

        idx = np.argsort(- ncc_scores)                 # descending
        best, second = ncc_scores[idx[0]], ncc_scores[idx[1]]

        # Convert similarity to distance and apply Lowe-style ratio test.
        best_dist = 1.0 - best
        second_dist = 1.0 - second
        if second_dist > 1e-8 and (best_dist / second_dist) < ratio_thresh and best > 0.0:
            matches.append((i, int(idx[0])))

    end = time.time()
    print(f"NCC Matching Time: {end - start:.3f}s — {len(matches)} matches")
    return matches
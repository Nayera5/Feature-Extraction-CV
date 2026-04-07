import numpy as np
import time

def match_ssd(descriptors_A, descriptors_B, ratio_thresh=0.75):
    matches = []
    
    start = time.time()
    
    for i, f in enumerate(descriptors_A):
        # Compute SSD with all descriptors in B
        diff = descriptors_B - f
        ssd = np.sum(diff**2, axis=1)
        
        # Get best and second best
        idx = np.argsort(ssd)
        best_idx = idx[0]
        second_idx = idx[1]
        
        best = ssd[best_idx]
        second = ssd[second_idx]
        
        # Ratio test
        if best / second < ratio_thresh:
            matches.append((i, best_idx))
    
    end = time.time()
    print("SSD Matching Time:", end - start)
    
    return matches
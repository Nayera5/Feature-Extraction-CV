import numpy as np
import time

def match_ncc(descriptors_A, descriptors_B, ratio_thresh=0.9):
    matches = []
    
    start = time.time()
    
    for i, f in enumerate(descriptors_A):
        # Normalize f
        f_mean = np.mean(f)
        f_std = np.std(f)
        f_norm = (f - f_mean) / (f_std + 1e-8)
        
        ncc_scores = []
        
        for g in descriptors_B:
            g_mean = np.mean(g)
            g_std = np.std(g)
            g_norm = (g - g_mean) / (g_std + 1e-8)
            
            ncc = np.sum(f_norm * g_norm)
            ncc_scores.append(ncc)
        
        ncc_scores = np.array(ncc_scores)
        
        # Best and second best
        idx = np.argsort(-ncc_scores)  # descending
        best_idx = idx[0]
        second_idx = idx[1]
        
        best = ncc_scores[best_idx]
        second = ncc_scores[second_idx]
        
        # Ratio-like test (since higher is better)
        if best > ratio_thresh and (best / (second + 1e-8)) > 1.1:
            matches.append((i, best_idx))
    
    end = time.time()
    print("NCC Matching Time:", end - start)
    
    return matches
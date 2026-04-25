"""
circle_detection.py — Hough transform for circle detection from scratch.

Implements the Hough transform to detect circles in edge maps.
"""

import numpy as np


def detect_circles(edges: np.ndarray,
                   threshold_percentage: float = 0.5) -> list:
    """
    Detect circles using the Hough transform (from scratch).

    Parameters
    ----------
    edges                : binary edge map (uint8, 0/255)
    min_radius           : minimum circle radius (default 10)
    max_radius           : maximum circle radius (default 100)
    threshold_percentage : percentage of max votes (0.0-1.0, default 0.9 = 90%)
    distance             : minimum distance between circle centers (default 15)

    Returns
    -------
    circles : list of (center_x, center_y, radius) tuples
    """
    min_radius = 10
    max_radius = 100
    distance = 15

    if edges.dtype != np.uint8:
        edges = (edges > 0).astype(np.uint8) * 255
    
    # Get edge pixels
    edge_points = np.where(edges > 0)
    if len(edge_points[0]) == 0:
        return []
    
    # Convert to (x, y) format
    points = np.column_stack((edge_points[1], edge_points[0]))
    h, w = edges.shape
    
    # Build accumulator for (x, y, r)
    accumulator = np.zeros((h, w, max_radius - min_radius + 1), dtype=np.uint32)
    
    # Precompute angles
    num_angles = 180
    angles = np.linspace(0, 2 * np.pi, num_angles, endpoint=False)
    cos_angles = np.cos(angles)
    sin_angles = np.sin(angles)
    
    # For each radius
    for r_idx, r in enumerate(range(min_radius, max_radius + 1)):
        # For each edge point
        for x, y in points:
            # For each angle
            # Calculate possible centers: (x - r*cos(θ), y - r*sin(θ))
            centers_x = x - r * cos_angles
            centers_y = y - r * sin_angles
            
            # Round to integer coordinates
            centers_x_int = np.round(centers_x).astype(int)
            centers_y_int = np.round(centers_y).astype(int)
            
            # Keep only valid coordinates
            valid = (centers_x_int >= 0) & (centers_x_int < w) & \
                    (centers_y_int >= 0) & (centers_y_int < h)
            
            valid_x = centers_x_int[valid]
            valid_y = centers_y_int[valid]
            
            # Accumulate votes
            accumulator[valid_y, valid_x, r_idx] += 1
    
    # Find peaks in accumulator
    max_votes = np.max(accumulator)
    adaptive_threshold = int(max_votes * threshold_percentage)
    circles = _find_circle_peaks(accumulator, min_radius, adaptive_threshold, distance, h, w)
        
    return circles


def _find_circle_peaks(accumulator, min_radius, threshold, distance, h, w):
    """Find circle peaks in the accumulator using non-maximum suppression."""
    circles = []
    
    # Get all peaks above threshold
    peaks = np.where(accumulator >= threshold)
    
    if len(peaks[0]) == 0:
        return circles
    
    # Create list of (y, x, r, votes)
    candidate_circles = []
    for y, x, r_idx in zip(peaks[0], peaks[1], peaks[2]):
        r = min_radius + r_idx
        votes = accumulator[y, x, r_idx]
        candidate_circles.append((y, x, r, votes))
    
    # Sort by votes descending
    candidate_circles.sort(key=lambda c: c[3], reverse=True)
    
    # Non-maximum suppression based on distance
    selected_circles = []
    for y, x, r, votes in candidate_circles:
        # Check if this circle overlaps with already selected ones
        is_duplicate = False
        for sy, sx, sr in selected_circles:
            # Distance between centers
            center_dist = np.sqrt((x - sx)**2 + (y - sy)**2)
            
            # If centers are within distance threshold, skip
            if center_dist < distance:
                is_duplicate = True
                break
        
        if not is_duplicate:
            selected_circles.append((y, x, r))
    
    # Convert to (x, y, r) format
    circles = [(x, y, r) for y, x, r in selected_circles]
    
    return circles

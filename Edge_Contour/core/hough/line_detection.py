import numpy as np

def detect_lines(edges: np.ndarray,
                 threshold_percentage: float = 0.5) -> list:

    height, width = edges.shape
    diag_len = int(np.sqrt(height**2 + width**2))

    rhos = np.arange(-diag_len, diag_len, 1)
    thetas = np.deg2rad(np.arange(0, 180))

    accumulator = np.zeros((len(rhos), len(thetas)), dtype=np.uint64)

    y_idxs, x_idxs = np.nonzero(edges)

    cos_t = np.cos(thetas)
    sin_t = np.sin(thetas)

    # Voting
    for i in range(len(x_idxs)):
        x = x_idxs[i]
        y = y_idxs[i]

        for t_idx in range(len(thetas)):
            rho = int(x * cos_t[t_idx] + y * sin_t[t_idx]) + diag_len
            accumulator[rho, t_idx] += 1

    # الآن بدل threshold عادي، نستخدم نسبة من max votes

    threshold = int(np.max(accumulator) * threshold_percentage)

    lines = []

    # لكل خلية في accumulator اللي عدت threshold
    for r in range(accumulator.shape[0]):
        for t in range(accumulator.shape[1]):
            votes = accumulator[r, t]
            if votes >= threshold and is_local_max(accumulator, r, t):

                rho = r - diag_len
                theta = thetas[t]

                cos_theta = np.cos(theta)
                sin_theta = np.sin(theta)

                # إيجاد النقاط اللي قريبة من الخط
                distance_threshold = 4
                points = []
                for i in range(len(x_idxs)):
                    x = x_idxs[i]
                    y = y_idxs[i]
                    dist = abs(x * cos_theta + y * sin_theta - rho)
                    if dist < distance_threshold:
                        points.append((x, y))

                if len(points) == 0:
                    continue

                points = np.array(points)
                # هنا نعمل normalization للـ votes على طول الخط

                # نقدر نستخدمه لتحديد إذا نحتفظ بالخط
                # مثلاً لو votes_normalized >= threshold_percentage * max_votes_normalized
                # لكن بما أننا استخدمنا already threshold_percentage على max_votes، فده يكفي

                # project each point onto the line direction
                dir_vec = np.array([-sin_theta, cos_theta])   # along-line unit vector
                projections = points @ dir_vec
                idx_min = np.argmin(projections)
                idx_max = np.argmax(projections)
                x1, y1 = points[idx_min]
                x2, y2 = points[idx_max]
                
                lines.append(((int(x1), int(y1)), (int(x2), int(y2))))

    return lines

def is_local_max(accumulator, r, t, window=3):
    r_min = max(r - window, 0)
    r_max = min(r + window + 1, accumulator.shape[0])
    t_min = max(t - window, 0)
    t_max = min(t + window + 1, accumulator.shape[1])

    return accumulator[r, t] == np.max(accumulator[r_min:r_max, t_min:t_max])


#-----------------------------------------------------------------------------------------

# import numpy as np
# from sklearn.cluster import DBSCAN

# def detect_lines(edges, threshold_percentage: float = 0.5) -> list:
#     """
#     Detect line segments like Probabilistic Hough.
    
#     edges: 2D uint8 array from Canny
#     min_line_length: minimum length of line segment to keep
#     max_gap: maximum gap between points to consider them connected
#     """
#     min_line_length=20
#     max_gap=5
#     # 1. Get all edge points
#     y_idxs, x_idxs = np.nonzero(edges)
#     points = np.array(list(zip(x_idxs, y_idxs)))

#     if len(points) == 0:
#         return []

#     # 2. Cluster connected points using DBSCAN
#     clustering = DBSCAN(eps=max_gap, min_samples=2).fit(points)
#     labels = clustering.labels_

#     segments = []

#     # 3. Fit a line to each cluster
#     for label in np.unique(labels):
#         if label == -1:
#             continue  # noise
#         cluster_points = points[labels == label]
#         if len(cluster_points) < min_line_length:
#             continue

#         # Fit line using linear regression (least squares)
#         x = cluster_points[:,0]
#         y = cluster_points[:,1]
#         A = np.vstack([x, np.ones(len(x))]).T
#         m, b = np.linalg.lstsq(A, y, rcond=None)[0]

#         # Find endpoints of the segment
#         x1, x2 = x.min(), x.max()
#         y1, y2 = int(m*x1 + b), int(m*x2 + b)

#         segments.append(((int(x1), int(y1)), (int(x2), int(y2))))

#     return segments

#------------------------------------------------------------------------------------------------

# import numpy as np
# import cv2

# def detect_lines(edges, threshold_percentage: float = 0.5) -> list:
#     """
#     Detect horizontal and vertical lines using projection + connected components.
    
#     Parameters
#     ----------
#     edges : 2D uint8 array (Canny edges)
#     min_line_length : minimum number of consecutive pixels to keep line
    
#     Returns
#     -------
#     lines : list of ((x1,y1),(x2,y2)) tuples
#     """
#     min_line_length = 10
#     lines = []

#     height, width = edges.shape

#     # ---------- Horizontal lines ----------
#     # Sum along columns -> projection on horizontal axis
#     horizontal_proj = np.sum(edges > 0, axis=1)
#     in_line = False
#     start_row = 0

#     for r in range(height):
#         if horizontal_proj[r] > 0 and not in_line:
#             in_line = True
#             start_row = r
#         elif horizontal_proj[r] == 0 and in_line:
#             end_row = r - 1
#             in_line = False
#             # Take only lines longer than min_line_length
#             if (end_row - start_row + 1) >= min_line_length:
#                 # For each horizontal line, find x start and end from edges
#                 cols = np.where(np.sum(edges[start_row:end_row+1, :], axis=0) > 0)[0]
#                 if len(cols) == 0:
#                     continue
#                 x1, x2 = cols[0], cols[-1]
#                 y1, y2 = start_row, end_row
#                 lines.append(((x1, y1), (x2, y2)))
#     # If still in line at bottom
#     if in_line:
#         end_row = height - 1
#         if (end_row - start_row + 1) >= min_line_length:
#             cols = np.where(np.sum(edges[start_row:end_row+1, :], axis=0) > 0)[0]
#             if len(cols) > 0:
#                 x1, x2 = cols[0], cols[-1]
#                 y1, y2 = start_row, end_row
#                 lines.append(((x1, y1), (x2, y2)))

#     # ---------- Vertical lines ----------
#     # Sum along rows -> projection on vertical axis
#     vertical_proj = np.sum(edges > 0, axis=0)
#     in_line = False
#     start_col = 0

#     for c in range(width):
#         if vertical_proj[c] > 0 and not in_line:
#             in_line = True
#             start_col = c
#         elif vertical_proj[c] == 0 and in_line:
#             end_col = c - 1
#             in_line = False
#             if (end_col - start_col + 1) >= min_line_length:
#                 rows = np.where(np.sum(edges[:, start_col:end_col+1], axis=1) > 0)[0]
#                 if len(rows) == 0:
#                     continue
#                 y1, y2 = rows[0], rows[-1]
#                 x1, x2 = start_col, end_col
#                 lines.append(((x1, y1), (x2, y2)))
#     # If still in line at right edge
#     if in_line:
#         end_col = width - 1
#         if (end_col - start_col + 1) >= min_line_length:
#             rows = np.where(np.sum(edges[:, start_col:end_col+1], axis=1) > 0)[0]
#             if len(rows) > 0:
#                 y1, y2 = rows[0], rows[-1]
#                 x1, x2 = start_col, end_col
#                 lines.append(((x1, y1), (x2, y2)))

#     return lines

# import numpy as np
# def detect_lines(edges, vote_threshold_percent=50):
#     """
#     Detect lines in an edge image using Hough transform.
    
#     Returns:
#     list: List of detected lines in (rho, theta, votes) format
#     """
#     height, width = edges.shape
#     diagonal = int(np.sqrt(height**2 + width**2))
    
#     # Rho range: -diagonal to diagonal with step 1
#     rho_range = np.arange(-diagonal, diagonal + 1, 1)
#     # Theta range: -90 to 90 degrees with step 1 degree
#     theta_range = np.arange(-90, 90, 1) * np.pi / 180
    
#     # Initialize the accumulator
#     accumulator = np.zeros((len(rho_range), len(theta_range)), dtype=np.int32)
    
#     # Mapping of (rho, theta) indices to (x, y) coordinates
#     edge_points = np.argwhere(edges > 0)
    
#     # Voting process
#     for y, x in edge_points:
#         for theta_idx, theta in enumerate(theta_range):
#             # Calculate rho = x*cos(theta) + y*sin(theta)
#             rho = int(x * np.cos(theta) + y * np.sin(theta))
#             # Map rho to its index in the accumulator
#             rho_idx = rho + diagonal
#             if 0 <= rho_idx < len(rho_range):
#                 accumulator[rho_idx, theta_idx] += 1
    
#     # Finding local maxima in the accumulator
#     detected_lines = []
    
#     # Calculate vote threshold based on percentage of maximum votes
#     max_votes = np.max(accumulator)
#     if max_votes == 0:
#         return []  # No lines detected
    
#     vote_threshold = int(max_votes * vote_threshold_percent / 100)
    
#     # Non-maxima suppression and threshold application
#     for rho_idx in range(1, len(rho_range) - 1):
#         for theta_idx in range(1, len(theta_range) - 1):
#             votes = accumulator[rho_idx, theta_idx]
            
#             # Check if it's above threshold
#             if votes > vote_threshold:
#                 is_local_max = True
                
#                 # Check 8-neighborhood for non-maxima suppression
#                 for dr in [-1, 0, 1]:
#                     for dt in [-1, 0, 1]:
#                         if dr == 0 and dt == 0:
#                             continue
#                         if accumulator[rho_idx + dr, theta_idx + dt] > votes:
#                             is_local_max = False
#                             break
#                     if not is_local_max:
#                         break
                
#                 if is_local_max:
#                     rho = rho_range[rho_idx]
#                     theta = theta_range[theta_idx]
#                     detected_lines.append((rho, theta, votes))
    
#     # Sort lines by vote count (descending)
#     detected_lines.sort(key=lambda x: x[2], reverse=True)
#     final_lines = merge_similar_lines(detected_lines)
    
#     return final_lines


# def merge_similar_lines(lines, rho_threshold=10, theta_threshold=np.pi/36):  # theta_threshold ≈ 5 degrees
#     if not lines:
#         return []
    
#     merged_lines = []
#     lines = sorted(lines, key=lambda x: x[2], reverse=True)  # Sort by votes
    
#     used = [False] * len(lines)
    
#     for i, (rho1, theta1, votes1) in enumerate(lines):
#         if used[i]:
#             continue
            
#         used[i] = True
#         similar_lines = [(rho1, theta1, votes1)]
        
#         for j, (rho2, theta2, votes2) in enumerate(lines[i+1:], i+1):
#             if used[j]:
#                 continue
                
#             # Check if lines are similar
#             if (abs(rho1 - rho2) < rho_threshold and 
#                 (abs(theta1 - theta2) < theta_threshold or 
#                  abs(abs(theta1 - theta2) - np.pi) < theta_threshold)):
#                 used[j] = True
#                 similar_lines.append((rho2, theta2, votes2))
        
#         # Average the parameters of similar lines, weighted by votes
#         total_votes = sum(line[2] for line in similar_lines)
#         avg_rho = sum(line[0] * line[2] for line in similar_lines) / total_votes
        
#         # Careful with theta averaging - need to handle wraparound
#         sin_avg = sum(np.sin(line[1]) * line[2] for line in similar_lines) / total_votes
#         cos_avg = sum(np.cos(line[1]) * line[2] for line in similar_lines) / total_votes
#         avg_theta = np.arctan2(sin_avg, cos_avg)
        
#         merged_lines.append((avg_rho, avg_theta, total_votes))
    
#     return merged_lines

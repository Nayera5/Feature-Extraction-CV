"""
Active Contour Model (Snake) with Greedy Algorithm
Evolves a contour to minimize energy through local neighbor search.
"""

from core.snake.energy import compute_image_energy, compute_internal_energy
import numpy as np



def greedy_snake(contour, image_energy, alpha=0.1, beta=0.1):
    """
    Returns:
        Updated contour with points moved to local minimum energy positions
    """
    new_contour = contour.copy()
    h, w = image_energy.shape

    for i in range(len(contour)):
        x, y = new_contour[i]  # Use new_contour to evolve with updated neighbors
        best_energy = float("inf")
        best_point = (x, y)

        # Search in 3x3 neighbourhood for minimum energy point
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                nx, ny = x + dx, y + dy
                
                # Boundary check: clamp coordinates
                nx = max(0, min(nx, w - 1))
                ny = max(0, min(ny, h - 1))
                candidate = (nx, ny)

                # Internal energy (elasticity + curvature)
                E_int = compute_internal_energy(new_contour, i, candidate, alpha, beta)
                # Image energy (attraction to edges)
                E_img = image_energy[ny, nx]
                # Total energy
                E = E_int + E_img

                if E < best_energy:
                    best_energy = E
                    best_point = candidate

        new_contour[i] = best_point

    return new_contour



def evolve_snake(image, contour, num_iterations=50, alpha=0.1, beta=0.1):
    """Evolve snake contour for multiple iterations.
    
    Args:
        image: Input image
        contour: Initial contour (array of (x, y) points)
        num_iterations: Number of evolution steps
        alpha: Elasticity weight (prefer nearby points)
        beta: Curvature weight (prefer smooth contours)
    
    Returns:
        Dictionary with:
            - 'contour': final evolved contour
            - 'history': list of contours at each iteration
    """
    # Compute image energy once
    image_energy = compute_image_energy(image)
    
    history = [contour.copy()]
    current_contour = contour.copy()
    
    for iteration in range(num_iterations):
        # Evolve contour by one step
        current_contour = greedy_snake(current_contour, image_energy, alpha, beta)
        history.append(current_contour.copy())
    
    return {
        'contour': current_contour,
        'history': history,
        'energy_map': image_energy
    }




def initialize_circular_contour(center, radius, num_points=50):
    """Initialize a circular contour around given center.
    
    Args:
        center: (cx, cy) center point
        radius: radius of circle
        num_points: number of points on the contour
    
    Returns:
        numpy array of shape (num_points, 2) with (x, y) coordinates
    """
    cx, cy = center
    angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
    x = cx + radius * np.cos(angles)
    y = cy + radius * np.sin(angles)
    contour = np.column_stack([x, y]).astype(int)
    return contour



# def initialize_rect_contour(top_left, width, height, num_points=50):
#     """Initialize a rectangular contour.
    
#     Args:
#         top_left: (x, y) top-left corner
#         width: rectangle width
#         height: rectangle height
#         num_points: total points on the perimeter
    
#     Returns:
#         numpy array of shape (num_points, 2) with (x, y) coordinates
#     """
#     x0, y0 = top_left
#     perimeter = 2 * (width + height)
#     points = []
    
#     # Top edge
#     top_count = int(num_points * width / perimeter)
#     for i in range(top_count):
#         x = x0 + int(i * width / top_count)
#         points.append([x, y0])
    
#     # Right edge
#     right_count = int(num_points * height / perimeter)
#     for i in range(right_count):
#         y = y0 + int(i * height / right_count)
#         points.append([x0 + width, y])
    
#     # Bottom edge
#     bottom_count = int(num_points * width / perimeter)
#     for i in range(bottom_count):
#         x = x0 + width - int(i * width / bottom_count)
#         points.append([x, y0 + height])
    
#     # Left edge
#     left_count = num_points - top_count - right_count - bottom_count
#     for i in range(left_count):
#         y = y0 + height - int(i * height / left_count)
#         points.append([x0, y])
    
#     return np.array(points, dtype=int)


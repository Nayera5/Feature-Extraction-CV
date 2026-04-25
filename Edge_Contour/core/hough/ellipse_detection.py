import cv2
import numpy as np

def detect_ellipses(image):
    """
    Detect ellipses in an image using Hough transform implementation.
        
    Returns:
        list: List of detected ellipses in 
    """
    # Convert to grayscale/color mask depending on image format.
    if image.ndim == 3:
        channels = image.shape[2]

        if channels == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
            channels = 3

        if channels == 3:
            # Convert to HSV for better color segmentation.
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            # Take saturation channel directly (avoid tuple-unpacking edge cases).
            saturation = hsv[:, :, 1]
            # Threshold on saturation to identify colored regions.
            _, mask = cv2.threshold(saturation, 20, 255, cv2.THRESH_BINARY)
        elif channels == 1:
            # Single-channel stored as HxWx1.
            gray = image[:, :, 0]
            _, mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
        else:
            raise ValueError(f"Unsupported channel count: {channels}")
    else:
        # If grayscale (HxW), use intensity thresholding.
        _, mask = cv2.threshold(image, 200, 255, cv2.THRESH_BINARY_INV)
    
    # Find contours in the mask
    # OpenCV 3 returns (image, contours, hierarchy), OpenCV 4 returns (contours, hierarchy).
    contours_result = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours_result) == 2:
        contours, _ = contours_result
    else:
        _, contours, _ = contours_result
    
    # Filter small contours
    min_area = 10
    contours = [c for c in contours if cv2.contourArea(c) > min_area]
    
    detected_ellipses = []
    
    for contour in contours:
        # We'll implement our own ellipse fitting algorithm here
        # 1. Calculate moments to find centroid
        M = cv2.moments(contour)
        if M["m00"] == 0:
            continue
        
        # Get centroid coordinates
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        
        # 2. Get points of the contour
        points = contour.reshape(-1, 2)
        
        # 3. Calculate covariance matrix for points
        # This gives us information about the shape and orientation
        points = points.astype(np.float64)
        points_centered = points - np.array([cx, cy])
        
        # Skip if too few points
        if len(points_centered) < 5:
            continue
            
        # Calculate covariance matrix
        cov = np.cov(points_centered.T)
        
        # 4. Get eigenvalues and eigenvectors of the covariance matrix
        # These give us the axes and orientation of the ellipse
        eigenvalues, eigenvectors = np.linalg.eig(cov)
        
        # Sort eigenvalues and corresponding eigenvectors
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # 5. Calculate major and minor axes
        # The eigenvalues are related to the variance along the axes
        major_axis = 2 * np.sqrt(5.991 * eigenvalues[0])  # 5.991 is chi-square value for 95% confidence
        minor_axis = 2 * np.sqrt(5.991 * eigenvalues[1])
        
        # Make sure we capture full extent of the blob
        # Find maximum distance from center to any contour point
        max_distance = 0
        for point in points:
            distance = np.sqrt((point[0] - cx)**2 + (point[1] - cy)**2)
            max_distance = max(max_distance, distance)
        
        # Adjust axes if needed
        scale_factor = 1.01 * max_distance / (major_axis / 2)
        major_axis *= scale_factor
        minor_axis *= scale_factor
        
        # 6. Calculate angle of orientation
        angle = np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]) * 180 / np.pi
        
        # Format as OpenCV ellipse: ((center_x, center_y), (major_axis, minor_axis), angle)
        ellipse = ((cx, cy), (major_axis, minor_axis), angle)
        detected_ellipses.append(ellipse)
    
    print(f"Detected {len(detected_ellipses)} ellipses")
    return detected_ellipses

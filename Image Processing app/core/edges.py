import numpy as np
import cv2
from core.image_manager import ImageManager


def sobel_edge_detection(image: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    
    # Convert to float for calculations
    image = image.astype(np.float64)
    
    # Sobel kernels
    # Kernel for detecting horizontal edges (gradient in x direction)
    sobel_x = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ], dtype=np.float64)
    
    # Kernel for detecting vertical edges (gradient in y direction)
    sobel_y = np.array([
        [-1, -2, -1],
        [ 0,  0,  0],
        [ 1,  2,  1]
    ], dtype=np.float64)
    
    # Apply convolution with Sobel kernels
    gradient_x = ImageManager.convolve(image, sobel_x)
    gradient_y = ImageManager.convolve(image, sobel_y)
    
    # Calculate gradient magnitude
    gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
    
    # Normalize to 0-255 range
    gradient_magnitude = (gradient_magnitude / gradient_magnitude.max() * 255).astype(np.uint8)
    
    return gradient_magnitude, gradient_x, gradient_y


def canny_edge_detection(image: np.ndarray, low_threshold: int = 50, high_threshold: int = 150) -> np.ndarray:
   
    return cv2.Canny(image, low_threshold, high_threshold)


def prewitt_edge_detection(image: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
   
    # Convert to float for calculations
    image = image.astype(np.float64)
    
    # Prewitt kernels (3x3)
    # Kernel for horizontal edges (gradient in x direction)
    prewitt_x = np.array([
        [-1, 0, 1],
        [-1, 0, 1],
        [-1, 0, 1]
    ], dtype=np.float64)
    
    # Kernel for vertical edges (gradient in y direction)
    prewitt_y = np.array([
        [-1, -1, -1],
        [ 0,  0,  0],
        [ 1,  1,  1]
    ], dtype=np.float64)
    
    # Apply convolution with Prewitt kernels
    gradient_x = ImageManager.convolve(image, prewitt_x)
    gradient_y = ImageManager.convolve(image, prewitt_y)
    
    # Calculate gradient magnitude
    gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
    
    # Normalize to 0-255 range
    if gradient_magnitude.max() > 0:
        gradient_magnitude = (gradient_magnitude / gradient_magnitude.max() * 255).astype(np.uint8)
    else:
        gradient_magnitude = gradient_magnitude.astype(np.uint8)
    
    return gradient_magnitude, gradient_x, gradient_y


def roberts_edge_detection(image: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    
    # Convert to float for calculations
    image = image.astype(np.float64)
    
    # Roberts Cross kernels (2x2)
    # These are applied differently - we handle the 2x2 case manually
    roberts_x = np.array([
        [1,  0],
        [0, -1]
    ], dtype=np.float64)
    
    roberts_y = np.array([
        [ 0, 1],
        [-1, 0]
    ], dtype=np.float64)
    
    # Apply convolution with Roberts kernels
    gradient_x = ImageManager.convolve(image, roberts_x)
    gradient_y = ImageManager.convolve(image, roberts_y)
    
    # Calculate gradient magnitude
    gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
    
    # Normalize to 0-255 range
    if gradient_magnitude.max() > 0:
        gradient_magnitude = (gradient_magnitude / gradient_magnitude.max() * 255).astype(np.uint8)
    else:
        gradient_magnitude = gradient_magnitude.astype(np.uint8)
    
    return gradient_magnitude, gradient_x, gradient_y

from core.canny.gradients import compute_gradients
import cv2

def compute_image_energy(image):

    # convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image

    gradient_magnitude = compute_gradients(gray)[0]
    gradient_magnitude = cv2.GaussianBlur(gradient_magnitude, (5,5), 0) # blur edges

    # image energy
    image_energy = - (gradient_magnitude) # (-) to minimise energy, we want to maximise gradient magnitude (edges)

    return image_energy



def compute_internal_energy(contour, i, candidate_point, alpha=0.1, beta=0.1):

    prev_point = contour[i-1]
    next_point = contour[(i+1) % len(contour)]

    # Elasticity (distance between points)
    elastic = (candidate_point[0] - prev_point[0])**2 + (candidate_point[1] - prev_point[1])**2

    # Smoothness
    curvature = (prev_point[0] - 2*candidate_point[0] + next_point[0])**2 + \
                (prev_point[1] - 2*candidate_point[1] + next_point[1])**2

    energy = alpha * elastic + beta * curvature

    return energy



import numpy as np
import cv2


def add_noise(image, noise_type, amount):
    if noise_type == "Gaussian":
        return gaussian_noise(image, amount)

    elif noise_type == "Uniform":
        return uniform_noise(image, amount)

    elif noise_type == "Salt & Pepper":
        return salt_pepper_noise(image, amount)

    return image


def gaussian_noise(image, amount):
    mean = 0
    sigma = amount * 50  # control strength
    gauss = np.random.normal(mean, sigma, image.shape).astype(np.float32)

    noisy = image.astype(np.float32) + gauss
    noisy = np.clip(noisy, 0, 255)

    return noisy.astype(np.uint8)


def uniform_noise(image, amount):
    low = -amount * 50
    high = amount * 50
    noise = np.random.uniform(low, high, image.shape)

    noisy = image.astype(np.float32) + noise
    noisy = np.clip(noisy, 0, 255)

    return noisy.astype(np.uint8)


def salt_pepper_noise(image, amount):
    noisy = image.copy()
    prob = amount

    rnd = np.random.rand(*image.shape[:2])

    noisy[rnd < prob / 2] = 0
    noisy[rnd > 1 - prob / 2] = 255

    return noisy
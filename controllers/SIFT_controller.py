from __future__ import annotations

from pathlib import Path
# import sys

import cv2
import numpy as np


# SIFT_DESCRIPTORS_DIR = Path(__file__).resolve().parents[1] / "sift descriptors"
# if str(SIFT_DESCRIPTORS_DIR) not in sys.path:
# 	sys.path.insert(0, str(SIFT_DESCRIPTORS_DIR))

from sift_detector.gaussian_pyramid import build_gaussian_pyramid
from sift_detector.dog_pyramid import build_dog_pyramid
from sift_detector.extrema_detection import detect_extrema, detect_extrema_fast
from sift_detector.keypoint_filter import filter_keypoints
from sift_descriptor import generate_descriptors


class SIFTController:
	def __init__(self, use_fast_extrema: bool = False):
		self.use_fast_extrema = use_fast_extrema

	def _to_gray_image(self, image: np.ndarray | str | Path) -> np.ndarray:
		if isinstance(image, (str, Path)):
			img = cv2.imread(str(image), cv2.IMREAD_GRAYSCALE)
			if img is None:
				raise ValueError(f"Could not read image: {image}")
			return img

		img = np.asarray(image)
		if img.ndim == 3:
			img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		return img

	def run(self, image: np.ndarray | str | Path) -> dict:
		gray = self._to_gray_image(image)

		gaussian_pyramid, gaussian_info = build_gaussian_pyramid(gray)
		dog_pyramid, dog_info = build_dog_pyramid(gaussian_pyramid)

		if self.use_fast_extrema:
			raw_keypoints, extrema_info = detect_extrema_fast(dog_pyramid)
		else:
			raw_keypoints, extrema_info = detect_extrema(dog_pyramid)

		filtered_keypoints, filter_info = filter_keypoints(raw_keypoints, dog_pyramid)
		descriptors, descriptor_info = generate_descriptors(filtered_keypoints, gaussian_pyramid)

		return {
			"gaussian_pyramid": gaussian_pyramid,
			"gaussian_info": gaussian_info,
			"dog_pyramid": dog_pyramid,
			"dog_info": dog_info,
			"raw_keypoints": raw_keypoints,
			"extrema_info": extrema_info,
			"filtered_keypoints": filtered_keypoints,
			"filter_info": filter_info,
			"descriptors": descriptors,
			"descriptor_info": descriptor_info,
		}

	def run_from_path(self, image_path: str | Path) -> dict:
		return self.run(image_path)

	def run_from_array(self, image: np.ndarray) -> dict:
		return self.run(image)

	def extract_descriptors(self, image: np.ndarray | str | Path):
		result = self.run(image)
		return result["descriptors"]


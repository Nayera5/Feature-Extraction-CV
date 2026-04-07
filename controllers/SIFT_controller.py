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
from matchers.ssd_matcher import match_ssd
from matchers.ncc_matcher import match_ncc


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

	def match_descriptors(self, descriptors_A, descriptors_B, technique: str = "ssd", ratio_thresh=None):
		"""
		Match descriptors from two images.
		
		Args:
			descriptors_A: List of descriptors from first image
			descriptors_B: List of descriptors from second image
			technique: Matching technique - "ssd" or "ncc" (default: "ssd")
			ratio_thresh: Threshold for Lowe's ratio test (default: 0.75 for SSD, 0.9 for NCC)
		
		Returns:
			List of tuples (idx_A, idx_B) representing matched descriptor pairs
		"""
		# Extract descriptor vectors from DescriptorKeypoint objects if needed
		desc_A = np.array([d.descriptor if hasattr(d, 'descriptor') else d for d in descriptors_A])
		desc_B = np.array([d.descriptor if hasattr(d, 'descriptor') else d for d in descriptors_B])
		
		if technique.lower() == "ssd":
			if ratio_thresh is None:
				ratio_thresh = 0.75
			return match_ssd(desc_A, desc_B, ratio_thresh=ratio_thresh)
		elif technique.lower() == "ncc":
			if ratio_thresh is None:
				ratio_thresh = 0.9
			return match_ncc(desc_A, desc_B, ratio_thresh=ratio_thresh)
		else:
			raise ValueError(f"Unknown matching technique: {technique}. Use 'ssd' or 'ncc'")

	def match_images(self, image1: np.ndarray | str | Path, image2: np.ndarray | str | Path, 
					 technique: str = "ssd", ratio_thresh=None):
		"""
		Extract descriptors from two images and match them.
		
		Args:
			image1: First image (path or array)
			image2: Second image (path or array)
			technique: Matching technique - "ssd" or "ncc" (default: "ssd")
			ratio_thresh: Threshold for matching
		
		Returns:
			Dictionary containing:
				- descriptors1: Descriptors from image 1
				- descriptors2: Descriptors from image 2
				- matches: List of matched pairs
		"""
		result1 = self.run(image1)
		result2 = self.run(image2)
		
		descriptors1 = result1["descriptors"]
		descriptors2 = result2["descriptors"]
		
		matches = self.match_descriptors(descriptors1, descriptors2, technique=technique, ratio_thresh=ratio_thresh)
		
		return {
			"descriptors1": descriptors1,
			"descriptors2": descriptors2,
			"matches": matches,
			"result1": result1,
			"result2": result2,
		}


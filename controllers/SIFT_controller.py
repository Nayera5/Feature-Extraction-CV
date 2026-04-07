from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from sift_detector.gaussian_pyramid import build_gaussian_pyramid
from sift_detector.dog_pyramid import build_dog_pyramid
from sift_detector.extrema_detection import detect_extrema_fast
from sift_detector.keypoint_filter import filter_keypoints
from sift_descriptor import generate_descriptors
from matchers.ssd_matcher import match_ssd , match_ncc


class SIFTController:
    def __init__(self, use_fast_extrema: bool = False):
        self.use_fast_extrema = use_fast_extrema

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _to_gray_image(self, image: np.ndarray | str | Path) -> np.ndarray:
        """Convert any supported input to a grayscale numpy array."""
        if isinstance(image, (str, Path)):
            img = cv2.imread(str(image), cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise ValueError(f"Could not read image: {image}")
            return img

        img = np.asarray(image)
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------

    def run(self, image: np.ndarray | str | Path) -> dict:
        """
        Full SIFT pipeline: pyramid → DoG → extrema → filter → descriptors.

        Returns a dict with all intermediate results plus:
            - filtered_keypoints
            - descriptors
        """
        gray = self._to_gray_image(image)

        gaussian_pyramid, gaussian_info = build_gaussian_pyramid(gray)
        dog_pyramid, dog_info = build_dog_pyramid(gaussian_pyramid)
        raw_keypoints, extrema_info = detect_extrema_fast(dog_pyramid)
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

    # ------------------------------------------------------------------
    # Matching — single entry point
    # ------------------------------------------------------------------

    def match_descriptors(
        self,
        descriptors_A,
        descriptors_B,
        technique: str = "ssd",
        ratio_thresh: float | None = None,
    ) -> list[tuple[int, int]]:
        """
        Match two descriptor lists using the chosen technique.

        Args:
            descriptors_A: descriptors from image A (list of DescriptorKeypoint or raw arrays)
            descriptors_B: descriptors from image B
            technique:     "ssd" or "ncc"
            ratio_thresh:  Lowe ratio threshold (default 0.75 for SSD, 0.9 for NCC)

        Returns:
            List of (idx_A, idx_B) matched pairs.
        """
        # Unwrap DescriptorKeypoint objects to plain numpy arrays if needed
        desc_A = np.array([d.descriptor if hasattr(d, "descriptor") else d for d in descriptors_A])
        desc_B = np.array([d.descriptor if hasattr(d, "descriptor") else d for d in descriptors_B])

        technique = technique.lower().strip()

        if technique == "ssd":
            return match_ssd(desc_A, desc_B, ratio_thresh=ratio_thresh or 0.55)
        elif technique == "ncc":
            return match_ncc(desc_A, desc_B, ratio_thresh=ratio_thresh or 0.75)
        else:
            raise ValueError(f"Unknown technique '{technique}'. Use 'ssd' or 'ncc'.")
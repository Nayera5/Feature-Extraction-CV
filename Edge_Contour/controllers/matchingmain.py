"""Matching Images controller for feature detection and matching."""

from __future__ import annotations

from pathlib import Path
import time

import cv2
import numpy as np
from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot, QThread
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QFileDialog, QMainWindow

from controllers.SIFT_controller import SIFTController


class MatchingWorker(QThread):
    """
    Background thread: runs the full SIFT + matching pipeline.

    Flow:
        run() image1  →  keypoints1, descriptors1
        run() image2  →  keypoints2, descriptors2
        match_descriptors(desc1, desc2, technique, ratio_thresh)  →  matches

    SIFTController.run() is the single extraction entry point.
    SIFTController.match_descriptors() is the single matching entry point.
    No other controller method is called here.
    """

    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(
        self,
        sift_controller: SIFTController,
        image1: np.ndarray,
        image2: np.ndarray,
        technique: str,
        ratio_thresh: float,
    ):
        super().__init__()
        self.sift_controller = sift_controller
        self.image1 = image1
        self.image2 = image2
        self.technique = technique
        self.ratio_thresh = ratio_thresh

    def run(self):
        try:
<<<<<<< match2
            t0 = time.perf_counter()

            # Step 1 — extract features independently for each image
            t_extract_start = time.perf_counter()
            result1 = self.sift_controller.run(cv2.cvtColor(self.image1, cv2.COLOR_RGB2BGR))
            result2 = self.sift_controller.run(cv2.cvtColor(self.image2, cv2.COLOR_RGB2BGR))
            extraction_sec = time.perf_counter() - t_extract_start
=======
            image1_prepared, image2_prepared = self.sift_controller.prepare_image_pair(self.image1, self.image2)
            result1 = self.sift_controller.run(image1_prepared)
            result2 = self.sift_controller.run(image2_prepared)
>>>>>>> main

            keypoints1   = result1["filtered_keypoints"]
            keypoints2   = result2["filtered_keypoints"]
            descriptors1 = result1["descriptors"]
            descriptors2 = result2["descriptors"]
<<<<<<< match2
=======
            filtered_keypoints1 = result1["filtered_keypoints"]
            filtered_keypoints2 = result2["filtered_keypoints"]
>>>>>>> main

            # Step 2 — match descriptors (single entry point)
            t_match_start = time.perf_counter()
            matches = self.sift_controller.match_descriptors(
                descriptors1,
                descriptors2,
                technique=self.technique,
                ratio_thresh=self.ratio_thresh,
            )
            matching_sec = time.perf_counter() - t_match_start
            total_sec = time.perf_counter() - t0

            self.finished.emit({
<<<<<<< match2
                "keypoints1":     keypoints1,
                "keypoints2":     keypoints2,
                "matches":        matches,
                "num_matches":    len(matches),
                "num_keypoints1": len(keypoints1),
                "num_keypoints2": len(keypoints2),

                "total_sec": total_sec,
=======
                "success": True,
                "keypoints1": descriptors1,
                "keypoints2": descriptors2,
                "descriptors1": descriptors1,
                "descriptors2": descriptors2,
                "matches": matches,
                "image1_prepared": image1_prepared,
                "image2_prepared": image2_prepared,
                "num_matches": len(matches),
                "num_keypoints1": len(filtered_keypoints1),
                "num_keypoints2": len(filtered_keypoints2),
>>>>>>> main
            })

        except Exception as e:
            self.error.emit(str(e))


class MatchingImagesController(QObject):
    """Controller for the Matching Images tab."""

    status_message = pyqtSignal(str)
    error_occurred = pyqtSignal(str)

    def __init__(self, parent: QMainWindow = None):
        super().__init__(parent)
        self.window = parent
        self.sift_controller = SIFTController(use_fast_extrema=False)
        self.match_viz_min_width = 600

<<<<<<< match2
        # Runtime state
        self.image1_array:       np.ndarray | None = None
        self.image2_array:       np.ndarray | None = None
        self.current_matches:    list | None = None
        self.current_keypoints1  = None
        self.current_keypoints2  = None
        self._worker:            MatchingWorker | None = None
=======
        # State
        self.image1_array = None
        self.image2_array = None
        self.current_matches = None
        self.current_keypoints1 = None
        self.current_keypoints2 = None
        self.current_image1_prepared = None
        self.current_image2_prepared = None
        self._worker = None
>>>>>>> main

    # ------------------------------------------------------------------
    # UI wiring — connect signals only, never touch widget geometry
    # ------------------------------------------------------------------

    def bind_ui(self, window: QMainWindow):
        self.window = window

        # Combo box items are defined in main.ui — never add them here
        self.window.matchingTechniqueCombo.setCurrentIndex(0)

        self.window.matchingLoadImage1Btn.clicked.connect(self.load_image_1)
        self.window.matchingLoadImage2Btn.clicked.connect(self.load_image_2)
        self.window.matchingRunBtn.clicked.connect(self.run_matching)
        
        # Connect spinbox value change to redraw matches (if results exist)
        self.window.matchingNumDisplaySpin.valueChanged.connect(self._on_num_display_changed)

        self._update_run_button()

    # ------------------------------------------------------------------
    # Image loading
    # ------------------------------------------------------------------

    def _pick_image(self) -> str:
        path, _ = QFileDialog.getOpenFileName(
            self.window,
            "Open Image",
            str(Path.home()),
            "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)",
        )
        return path

    def _load_image(self, slot: int):
        path = self._pick_image()
        if not path:
            return
        try:
            raw = cv2.imread(path)
            if raw is None:
                raise ValueError("Could not decode image file.")
            image_rgb = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)

            if slot == 1:
                self.image1_array = raw
                self._display_preview(self.window.matchingImage1Host, image_rgb)
                self.window.matchingImage1NameLbl.setText(Path(path).name)
            else:
                self.image2_array = raw
                self._display_preview(self.window.matchingImage2Host, image_rgb)
                self.window.matchingImage2NameLbl.setText(Path(path).name)

            self.status_message.emit(f"Loaded Image {slot}: {Path(path).name}")
            self._update_run_button()

        except Exception as e:
            self.error_occurred.emit(f"Error loading Image {slot}: {e}")

    def load_image_1(self):
        self._load_image(1)

    def load_image_2(self):
        self._load_image(2)

    # ------------------------------------------------------------------
    # Display helpers — sizes come from the label's actual geometry
    # ------------------------------------------------------------------

    @staticmethod
    def _fit_image_to_label(label, image_rgb: np.ndarray) -> QPixmap:
        """Scale image_rgb to fit label dimensions, preserving aspect ratio."""
        lw = label.width()  or 400
        lh = label.height() or 280

        h, w = image_rgb.shape[:2]
        scale = min(lw / w, lh / h, 1.0)   # never upscale past native resolution
        if scale < 1.0:
            w = int(w * scale)
            h = int(h * scale)
            image_rgb = cv2.resize(image_rgb, (w, h), interpolation=cv2.INTER_AREA)

        q_img = QImage(image_rgb.data, w, h, 3 * w, QImage.Format_RGB888)
        return QPixmap.fromImage(q_img)

    def _display_preview(self, label, image_rgb: np.ndarray):
        try:
            label.setPixmap(self._fit_image_to_label(label, image_rgb))
        except Exception as e:
            print(f"[preview] {e}")

    def _display_image_in_label(self, label, image_rgb: np.ndarray):
        try:
            h, w = image_rgb.shape[:2]
            q_img = QImage(image_rgb.data, w, h, 3 * w, QImage.Format_RGB888).copy()
            pixmap = QPixmap.fromImage(q_img)
            label.setPixmap(pixmap)
            label.resize(pixmap.size())
        except Exception as e:
            print(f"[visualization] {e}")

    # ------------------------------------------------------------------
    # Matching — kicks off the worker thread
    # ------------------------------------------------------------------

    def _update_run_button(self):
        both_loaded = self.image1_array is not None and self.image2_array is not None
        self.window.matchingRunBtn.setEnabled(both_loaded)

    def _on_num_display_changed(self):
        """Redraw matches when user changes the display count spinner."""
        if self.current_matches is not None:
            self._draw_matches()

    def run_matching(self):
        """Validate inputs and launch the background MatchingWorker."""
        if self.image1_array is None or self.image2_array is None:
            self.error_occurred.emit("Both images must be loaded before matching.")
            return

        if self._worker and self._worker.isRunning():
            self.status_message.emit("Matching already in progress…")
            return

        technique = self.window.matchingTechniqueCombo.currentText().lower().strip()
        if technique not in ("ssd", "ncc"):
            self.error_occurred.emit(f"Unknown technique '{technique}'. Expected SSD or NCC.")
            return

        ratio_thresh = self.window.matchingRatioSpin.value()

        self.window.matchingRunBtn.setEnabled(False)
        self.status_message.emit(f"Running SIFT + {technique.upper()} matching…")

        self._worker = MatchingWorker(
            self.sift_controller,
            self.image1_array,
            self.image2_array,
            technique,
            ratio_thresh,
        )
        self._worker.finished.connect(self._on_matching_finished)
        self._worker.error.connect(self._on_matching_error)
        self._worker.start()

    # ------------------------------------------------------------------
    # Worker callbacks
    # ------------------------------------------------------------------

    @pyqtSlot(dict)
    def _on_matching_finished(self, result: dict):
        try:
            self.current_matches    = result["matches"]
            self.current_keypoints1 = result["keypoints1"]
            self.current_keypoints2 = result["keypoints2"]
            self.current_image1_prepared = result["image1_prepared"]
            self.current_image2_prepared = result["image2_prepared"]

            self.window.matchingStatsLbl.setText(
                f"Keypoints Image 1: {result['num_keypoints1']}\n"
                f"Keypoints Image 2: {result['num_keypoints2']}\n"
                f"Matches Found:     {result['num_matches']}\n"
                f"Total Time:        {result['total_sec']:.3f} s"
            )
            self._draw_matches()
            self.status_message.emit(
                f"Done: {result['num_matches']} matches in {result['total_sec']:.3f}s."
            )

        except Exception as e:
            self.error_occurred.emit(f"Result processing error: {e}")
        finally:
            self.window.matchingRunBtn.setEnabled(True)

    @pyqtSlot(str)
    def _on_matching_error(self, msg: str):
        self.error_occurred.emit(f"Matching failed: {msg}")
        self.window.matchingRunBtn.setEnabled(True)

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------

    def _draw_matches(self):
        """Render side-by-side images with keypoint circles and match lines."""
        if not self.current_matches:
            self.window.matchingVisualizationHost.setText("No matches found.")
            return

        try:
<<<<<<< match2
            img1 = cv2.cvtColor(self.image1_array, cv2.COLOR_RGB2BGR)
            img2 = cv2.cvtColor(self.image2_array, cv2.COLOR_RGB2BGR)

            h1, w1 = img1.shape[:2]
            h2, w2 = img2.shape[:2]
            max_h  = max(h1, h2)

            # Pad the shorter image so hstack produces a clean rectangle
            if h1 < max_h:
                img1 = cv2.copyMakeBorder(img1, 0, max_h - h1, 0, 0, cv2.BORDER_CONSTANT)
            if h2 < max_h:
                img2 = cv2.copyMakeBorder(img2, 0, max_h - h2, 0, 0, cv2.BORDER_CONSTANT)
=======
            img1 = self.current_image1_prepared.copy()
            img2 = self.current_image2_prepared.copy()
            w1 = img1.shape[1]
>>>>>>> main

            combined = np.hstack([img1, img2])

            # Get user-selected number of matches to display
            num_to_display = self.window.matchingNumDisplaySpin.value()
            num_to_show = min(num_to_display, len(self.current_matches))

            for idx_a, idx_b in self.current_matches[:num_to_show]:
                kp1 = self.current_keypoints1[idx_a]
                kp2 = self.current_keypoints2[idx_b]

                pt1 = (int(kp1.x), int(kp1.y)) if hasattr(kp1, "x") else (int(kp1[0]), int(kp1[1]))
                pt2 = (int(kp2.x), int(kp2.y)) if hasattr(kp2, "x") else (int(kp2[0]), int(kp2[1]))
                pt2_shifted = (pt2[0] + w1, pt2[1])

                cv2.circle(combined, pt1,         4, (0, 255, 0), -1)
                cv2.circle(combined, pt2_shifted, 4, (0, 255, 0), -1)
                cv2.line(combined, pt1, pt2_shifted, (255, 100, 0), 1)

            # Enlarge the output image itself (not only the container) for clearer inspection.
            h_combined, w_combined = combined.shape[:2]
            if w_combined > 0 and w_combined < self.match_viz_min_width:
                scale = self.match_viz_min_width / float(w_combined)
                new_w = int(w_combined * scale)
                new_h = int(h_combined * scale)
                combined = cv2.resize(combined, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

            combined_rgb = cv2.cvtColor(combined, cv2.COLOR_BGR2RGB)
            self._display_image_in_label(self.window.matchingVisualizationHost, combined_rgb)

        except Exception as e:
            self.window.matchingVisualizationHost.setText(f"Visualization error: {e}")
            print(f"[matches viz] {e}")
"""Matching Images controller for feature detection and matching."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot, QThread, Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QFileDialog, QMainWindow

from controllers.SIFT_controller import SIFTController


class MatchingWorker(QThread):
    """Background worker thread for matching operations."""

    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, sift_controller, image1, image2, technique: str, ratio_thresh: float):
        super().__init__()
        self.sift_controller = sift_controller
        self.image1 = image1
        self.image2 = image2
        self.technique = technique
        self.ratio_thresh = ratio_thresh

    def run(self):
        try:
            result1 = self.sift_controller.run(self.image1)
            result2 = self.sift_controller.run(self.image2)

            descriptors1 = result1["descriptors"]
            descriptors2 = result2["descriptors"]
            keypoints1 = result1["filtered_keypoints"]
            keypoints2 = result2["filtered_keypoints"]

            matches = self.sift_controller.match_descriptors(
                descriptors1,
                descriptors2,
                technique=self.technique,
                ratio_thresh=self.ratio_thresh,
            )

            self.finished.emit({
                "success": True,
                "keypoints1": keypoints1,
                "keypoints2": keypoints2,
                "descriptors1": descriptors1,
                "descriptors2": descriptors2,
                "matches": matches,
                "num_matches": len(matches),
                "num_keypoints1": len(keypoints1),
                "num_keypoints2": len(keypoints2),
            })
        except Exception as e:
            self.error.emit(str(e))


class MatchingImagesController(QObject):
    """Controller for image matching functionality."""

    status_message = pyqtSignal(str)
    error_occurred = pyqtSignal(str)

    def __init__(self, parent: QMainWindow = None):
        super().__init__(parent)
        self.window = parent
        self.sift_controller = SIFTController(use_fast_extrema=False)

        # State
        self.image1_array = None
        self.image2_array = None
        self.current_matches = None
        self.current_keypoints1 = None
        self.current_keypoints2 = None
        self._worker = None

    # ------------------------------------------------------------------
    # UI wiring
    # ------------------------------------------------------------------

    def bind_ui(self, window: QMainWindow):
        """Bind UI elements to controller methods.
        
        All widget sizes/properties are defined in main.ui.
        This method only connects signals — it never mutates widget geometry.
        """
        self.window = window

        # Combo box items are defined in main.ui; never add them here.
        # Just ensure a valid default selection.
        self.window.matchingTechniqueCombo.setCurrentIndex(0)

        self.window.matchingLoadImage1Btn.clicked.connect(self.load_image_1)
        self.window.matchingLoadImage2Btn.clicked.connect(self.load_image_2)
        self.window.matchingRunBtn.clicked.connect(self.match_images)

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
        """Generic loader for either image slot (1 or 2)."""
        path = self._pick_image()
        if not path:
            return

        try:
            raw = cv2.imread(path)
            if raw is None:
                raise ValueError("Could not decode image file")
            image_rgb = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)

            if slot == 1:
                self.image1_array = image_rgb
                self._display_preview(self.window.matchingImage1Host, image_rgb)
                self.window.matchingImage1NameLbl.setText(Path(path).name)
            else:
                self.image2_array = image_rgb
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
    # Display helpers — no hardcoded pixel sizes; use the label's actual size
    # ------------------------------------------------------------------

    @staticmethod
    def _fit_image_to_label(label, image_rgb: np.ndarray) -> QPixmap:
        """Scale image_rgb to fit inside label while preserving aspect ratio."""
        lw = label.width() or 400   # fallback if layout not yet realized
        lh = label.height() or 280

        h, w = image_rgb.shape[:2]
        scale = min(lw / w, lh / h, 1.0)   # never upscale beyond native size
        if scale < 1.0:
            w, h = int(w * scale), int(h * scale)
            image_rgb = cv2.resize(image_rgb, (w, h), interpolation=cv2.INTER_AREA)

        q_img = QImage(image_rgb.data, w, h, 3 * w, QImage.Format_RGB888)
        return QPixmap.fromImage(q_img)

    def _display_preview(self, label, image_rgb: np.ndarray):
        """Show a scaled preview inside a QLabel."""
        try:
            pixmap = self._fit_image_to_label(label, image_rgb)
            label.setPixmap(pixmap)
        except Exception as e:
            print(f"[preview] {e}")

    def _display_image_in_label(self, label, image_rgb: np.ndarray):
        """Show the full visualization image inside a QLabel."""
        try:
            pixmap = self._fit_image_to_label(label, image_rgb)
            label.setPixmap(pixmap)
        except Exception as e:
            print(f"[visualization] {e}")

    # ------------------------------------------------------------------
    # Matching
    # ------------------------------------------------------------------

    def _update_run_button(self):
        both = self.image1_array is not None and self.image2_array is not None
        self.window.matchingRunBtn.setEnabled(both)

    def match_images(self):
        if self.image1_array is None or self.image2_array is None:
            self.error_occurred.emit("Both images must be loaded before matching.")
            return

        if self._worker and self._worker.isRunning():
            self.status_message.emit("Matching already in progress…")
            return

        # Read technique from combo box (items come from .ui, not from code)
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

    @pyqtSlot(dict)
    def _on_matching_finished(self, result: dict):
        try:
            self.current_matches = result["matches"]
            self.current_keypoints1 = result["keypoints1"]
            self.current_keypoints2 = result["keypoints2"]

            self.window.matchingStatsLbl.setText(
                f"Keypoints Image 1: {result['num_keypoints1']}\n"
                f"Keypoints Image 2: {result['num_keypoints2']}\n"
                f"Matches Found:     {result['num_matches']}"
            )

            self._display_matches()
            self.status_message.emit(f"Done — {result['num_matches']} matches found.")
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

    def _display_matches(self):
        """Draw matched keypoints on a side-by-side image and show it."""
        if not self.current_matches:
            self.window.matchingVisualizationHost.setText("No matches found.")
            return

        try:
            img1 = cv2.cvtColor(self.image1_array, cv2.COLOR_RGB2BGR)
            img2 = cv2.cvtColor(self.image2_array, cv2.COLOR_RGB2BGR)

            # Pad shorter image vertically so hstack works cleanly
            h1, w1 = img1.shape[:2]
            h2, w2 = img2.shape[:2]
            max_h = max(h1, h2)
            if h1 < max_h:
                img1 = cv2.copyMakeBorder(img1, 0, max_h - h1, 0, 0, cv2.BORDER_CONSTANT)
            if h2 < max_h:
                img2 = cv2.copyMakeBorder(img2, 0, max_h - h2, 0, 0, cv2.BORDER_CONSTANT)

            combined = np.hstack([img1, img2])

            for idx_a, idx_b in self.current_matches[:20]:
                kp1 = self.current_keypoints1[idx_a]
                kp2 = self.current_keypoints2[idx_b]

                pt1 = (int(kp1.x), int(kp1.y)) if hasattr(kp1, "x") else (int(kp1[0]), int(kp1[1]))
                pt2 = (int(kp2.x), int(kp2.y)) if hasattr(kp2, "x") else (int(kp2[0]), int(kp2[1]))
                pt2_shifted = (pt2[0] + w1, pt2[1])

                cv2.circle(combined, pt1, 4, (0, 255, 0), -1)
                cv2.circle(combined, pt2_shifted, 4, (0, 255, 0), -1)
                cv2.line(combined, pt1, pt2_shifted, (255, 100, 0), 1)

            combined_rgb = cv2.cvtColor(combined, cv2.COLOR_BGR2RGB)
            self._display_image_in_label(self.window.matchingVisualizationHost, combined_rgb)

        except Exception as e:
            self.window.matchingVisualizationHost.setText(f"Visualization error: {e}")
            print(f"[matches viz] {e}")
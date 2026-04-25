"""Hough controller for shape detection (lines, circles, etc)."""

from __future__ import annotations

import numpy as np
from pathlib import Path

from PyQt5.QtCore import QEvent, QObject, Qt, QThread, pyqtSignal
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QLabel

from controllers.main_controller import MainController
from core.hough import (
    run_hough_line_detection,
    run_hough_circle_detection,
    run_hough_ellipse_detection,
)


class _HoughImageInteractor(QObject):
    """Manages image display in a label."""

    def __init__(self, label: QLabel, pixmap_factory):
        super().__init__(label)
        self._label = label
        self._pixmap_factory = pixmap_factory
        self._pixmap_orig: QPixmap | None = None
        self._label.setAlignment(Qt.AlignCenter)

    def set_array(self, array: np.ndarray):
        self._pixmap_orig = self._pixmap_factory(array)
        self._display_scaled()

    def _display_scaled(self):
        if self._pixmap_orig is None:
            return
        # Scale uniformly to fit label size while keeping aspect ratio
        scaled = self._pixmap_orig.scaledToWidth(
            min(self._label.width(), self._label.height()),
            Qt.SmoothTransformation,
        )
        self._label.setPixmap(scaled)
        self._label.setText("")


class _HoughWorker(QThread):
    """Runs the shape detection pipeline in a background thread."""

    finished = pyqtSignal(dict)  # emits the full result dict
    error = pyqtSignal(str)

    def __init__(self, image: np.ndarray, params: dict, detection_type: str = "line", parent=None):
        super().__init__(parent)
        self._image = image
        self._params = params
        self._detection_type = detection_type

    def run(self):
        try:
            if self._detection_type == "circle":
                result = run_hough_circle_detection(
                    self._image,
                    kernel_size=self._params.get("kernel_size", 3),
                    sigma=self._params.get("sigma", 1.4),
                    low_ratio=self._params.get("low_ratio", 0.05),
                    high_ratio=self._params.get("high_ratio", 0.15),
                    threshold_percentage=self._params.get("threshold_percentage", 0.9),
                )
            elif self._detection_type == "ellipse":
                result = run_hough_ellipse_detection(
                    self._image,
                    kernel_size=self._params.get("kernel_size", 3),
                    sigma=self._params.get("sigma", 1.4),
                    low_ratio=self._params.get("low_ratio", 0.05),
                    high_ratio=self._params.get("high_ratio", 0.15),
                    threshold_percentage=self._params.get("threshold_percentage", 0.9),
                )
            else:  # line
                result = run_hough_line_detection(
                    self._image,
                    kernel_size=self._params.get("kernel_size", 3),
                    sigma=self._params.get("sigma", 1.4),
                    low_ratio=self._params.get("low_ratio", 0.05),
                    high_ratio=self._params.get("high_ratio", 0.15),
                    threshold_percentage=self._params.get("threshold_percentage", 0.9),
                )
            self.finished.emit(result)
        except Exception as exc:  # noqa: BLE001
            self.error.emit(str(exc))


class HoughController(MainController):
    """Controller for Hough-based shape detection."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._ui = None
        self._current_image: np.ndarray | None = None
        self._results: dict | None = None
        self._image_view: _HoughImageInteractor | None = None

    def bind_ui(self, ui):
        """Wire up UI elements for shape detection."""
        self._ui = ui
        self._image_view = _HoughImageInteractor(
            ui.shapesImageHost, self.to_qpixmap_image
        )

        # Image loading
        ui.shapesOpenBtn.clicked.connect(self.open_image_dialog)
        ui.shapesDetectBtn.clicked.connect(self.run_from_ui)
        ui.shapesSaveBtn.clicked.connect(self.save_result_from_ui)
        
        # Detection type combo box
        ui.shapesDetectionTypeCombo.currentTextChanged.connect(self._on_detection_type_changed)
        self._on_detection_type_changed(ui.shapesDetectionTypeCombo.currentText())

        # Connect result signals
        self.result_ready.connect(self._handle_result)

    def open_image_dialog(self):
        if self._ui is None:
            return
        path = self.pick_image_path(self._ui)
        if path:
            self.load_image(path)

    def load_image(self, path: str):
        if self._ui is None or self._image_view is None:
            return
        try:
            self._current_image = self.load_image_array(path)
        except Exception as exc:  # noqa: BLE001
            self.error_occurred.emit(str(exc))
            return

        self._ui.shapesImageNameLbl.setText(Path(path).name)
        self._results = None
        self._image_view.set_array(self._current_image)
        self._ui.shapesDetectBtn.setEnabled(True)
        self._ui.shapesSaveBtn.setEnabled(False)
        self._ui.shapesStatsLbl.setText(
            f"Size: {self._current_image.shape[1]} × {self._current_image.shape[0]}\n"
            f"Channels: {self._current_image.ndim}"
        )

    def run_from_ui(self):
        if self._ui is None or self._current_image is None:
            return
        
        detection_type = self._ui.shapesDetectionTypeCombo.currentText()
        
        # Convert to lowercase for internal use
        if "Line" in detection_type:
            detection_type_key = "line"
        elif "Circle" in detection_type:
            detection_type_key = "circle"
        else:  # Ellipse
            detection_type_key = "ellipse"
        
        params = {
            "kernel_size": self._ui.shapesKernelSpin.value(),
            "sigma": self._ui.shapesSigmaSpin.value(),
            "low_ratio": self._ui.shapesLowSpin.value(),
            "high_ratio": self._ui.shapesHighSpin.value(),
            "threshold_percentage": self._ui.shapesThresholdSpin.value() / 100.0,
            "min_radius": 10,
            "max_radius": 100,
            "distance": 15,
            "min_semi_major": 15,
            "max_semi_major": 150,
        }
        self._ui.shapesDetectBtn.setEnabled(False)
        self.process_image_array(self._current_image, params, detection_type_key)

    def save_result_from_ui(self):
        if self._ui is None or self._results is None:
            return
        from PyQt5.QtWidgets import QFileDialog

        path, _ = QFileDialog.getSaveFileName(
            self._ui,
            "Save Detection Result",
            "line_detection.png",
            "PNG (*.png);;JPEG (*.jpg);;BMP (*.bmp)",
        )
        if path:
            self.save_image_array(self._results["marked_image"], path)

    def process_image_array(self, image: np.ndarray, params: dict, detection_type: str = "line"):
        """Process image with shape detection."""
        self._run(image, params, detection_type)

    def _run(self, image: np.ndarray, params: dict, detection_type: str = "line"):
        worker = _HoughWorker(image, params, detection_type, parent=self)
        self._start_worker(worker, f"Running {detection_type.capitalize()} detection...")

    def _on_detection_type_changed(self, detection_type_text: str):
        """Update UI labels based on selected detection type."""
        if self._ui is None:
            return
        
        # Update the action text; threshold is shared across detection types.
        if "Line" in detection_type_text:
            self._ui.shapesDetectBtn.setText("▶  Detect Lines")
        elif "Circle" in detection_type_text:
            self._ui.shapesDetectBtn.setText("▶  Detect Circles")
        else:  # Ellipse
            self._ui.shapesDetectBtn.setText("▶  Detect Ellipses")

    def _on_finished(self, result: dict):
        super()._on_finished(result)
        # Determine which type of shape was detected
        if "lines" in result:
            self.status_message.emit("Line detection complete.")
        elif "circles" in result:
            self.status_message.emit("Circle detection complete.")
        else:
            self.status_message.emit("Ellipse detection complete.")

    def _handle_result(self, result: dict):
        if self._ui is None or self._image_view is None:
            return

        self._results = result
        
        # Display the image with detected shapes drawn on it
        self._image_view.set_array(result["marked_image"])
        
        # Update stats based on detected shapes
        if not hasattr(self._ui, 'shapesStatsLbl'):
            return
            
        if "lines" in result:
            shapes = result.get("lines", [])
            self._ui.shapesStatsLbl.setText(
                f"Lines detected: {len(shapes)}\n"
                f"Result size: {result['marked_image'].shape[1]} × {result['marked_image'].shape[0]}"
            )
        elif "circles" in result:
            shapes = result.get("circles", [])
            self._ui.shapesStatsLbl.setText(
                f"Circles detected: {len(shapes)}\n"
                f"Result size: {result['marked_image'].shape[1]} × {result['marked_image'].shape[0]}"
            )
        else:  # ellipses
            shapes = result.get("ellipses", [])
            self._ui.shapesStatsLbl.setText(
                f"Ellipses detected: {len(shapes)}\n"
                f"Result size: {result['marked_image'].shape[1]} × {result['marked_image'].shape[0]}"
            )
        
        if hasattr(self._ui, 'shapesDetectBtn'):
            self._ui.shapesDetectBtn.setEnabled(True)
        if hasattr(self._ui, 'shapesSaveBtn'):
            self._ui.shapesSaveBtn.setEnabled(True)

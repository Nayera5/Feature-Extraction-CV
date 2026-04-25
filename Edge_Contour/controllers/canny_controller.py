"""Canny controller bound directly to the widgets defined in main.ui."""

from __future__ import annotations

import numpy as np
from pathlib import Path

from PyQt5.QtCore import QEvent, QObject, Qt, QThread, pyqtSignal
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QLabel

from controllers.main_controller import MainController
from core.canny import run_canny


class _CannyImageInteractor(QObject):
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


class _CannyWorker(QThread):
    """Runs the (slow) Canny pipeline in a background thread."""

    finished  = pyqtSignal(dict)      # emits the full result dict
    error     = pyqtSignal(str)

    def __init__(self, image: np.ndarray, params: dict, parent=None):
        super().__init__(parent)
        self._image  = image
        self._params = params

    def run(self):
        try:
            result = run_canny(
                self._image,
                kernel_size=self._params.get("kernel_size", 5),
                sigma      =self._params.get("sigma",       1.4),
                low_ratio  =self._params.get("low_ratio",   0.05),
                high_ratio =self._params.get("high_ratio",  0.15),
            )
            self.finished.emit(result)
        except Exception as exc:                        # noqa: BLE001
            self.error.emit(str(exc))


class CannyController(MainController):
    STAGES = [
        ("Original", "original"),
        ("Grayscale", "gray"),
        ("Blurred", "blurred"),
        ("Magnitude", "magnitude"),
        ("Suppressed", "suppressed"),
        ("Thresholded", "thresholded"),
        ("Edges", "edges"),
    ]

    def __init__(self, parent=None):
        super().__init__(parent)
        self._ui = None
        self._current_image: np.ndarray | None = None
        self._results: dict | None = None
        self._image_view: _CannyImageInteractor | None = None

    def bind_ui(self, ui):
        self._ui = ui
        self._image_view = _CannyImageInteractor(ui.cannyImageHost, self.to_qpixmap_image)

        ui.cannyStageCombo.clear()
        for label, _ in self.STAGES:
            ui.cannyStageCombo.addItem(label)
        ui.cannyStageCombo.setCurrentIndex(0)

        ui.cannyOpenBtn.clicked.connect(self.open_image_dialog)
        ui.cannyRunBtn.clicked.connect(self.run_from_ui)
        ui.cannySaveBtn.clicked.connect(self.save_edges_from_ui)
        ui.cannyStageCombo.currentIndexChanged.connect(self.on_stage_changed)

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

        self._ui.cannyImageNameLbl.setText(Path(path).name)
        self._results = None
        self._ui.cannyStageCombo.setCurrentIndex(0)
        self._image_view.set_array(self._current_image)
        self._ui.cannyRunBtn.setEnabled(True)
        self._ui.cannySaveBtn.setEnabled(False)
        self._ui.cannyStatsLbl.setText(
            f"Size: {self._current_image.shape[1]} × {self._current_image.shape[0]}\n"
            f"Channels: {self._current_image.ndim}"
        )

    def run_from_ui(self):
        if self._ui is None or self._current_image is None:
            return
        params = {
            "kernel_size": self._ui.cannyKernelSpin.value(),
            "sigma": self._ui.cannySigmaSpin.value(),
            "low_ratio": self._ui.cannyLowSpin.value(),
            "high_ratio": self._ui.cannyHighSpin.value(),
        }
        self._ui.cannyRunBtn.setEnabled(False)
        self.process_image_array(self._current_image, params)

    def save_edges_from_ui(self):
        if self._ui is None or self._results is None:
            return
        from PyQt5.QtWidgets import QFileDialog

        path, _ = QFileDialog.getSaveFileName(
            self._ui,
            "Save Edge Map",
            "edge_map.png",
            "PNG (*.png);;JPEG (*.jpg);;BMP (*.bmp)",
        )
        if path:
            self.save_image_array(self._results["edges"], path)

    def on_stage_changed(self, idx: int):
        if self._ui is None or self._image_view is None:
            return

        if self._results is None and idx == 0 and self._current_image is not None:
            self._image_view.set_array(self._current_image)
            return
        if self._results is None:
            return

        _, key = self.STAGES[idx]
        array = self._results.get(key)
        if array is not None:
            self._image_view.set_array(array)

    # ── public slots (called by UI) ─────────────────────────────────────────

    def process_image_path(self, path: str, params: dict):
        """Load *path* from disk then run the pipeline with *params*."""
        try:
            image = self.load_image_array(path)
        except Exception as exc:                        # noqa: BLE001
            self.error_occurred.emit(f"Failed to load image: {exc}")
            return
        self._run(image, params)

    def process_image_array(self, image: np.ndarray, params: dict):
        self._run(image, params)

    # ── internal helpers ────────────────────────────────────────────────────

    def _run(self, image: np.ndarray, params: dict):
        worker = _CannyWorker(image, params, parent=self)
        self._start_worker(worker, "Running Canny edge detection...")

    def _on_finished(self, result: dict):
        super()._on_finished(result)
        self.status_message.emit("Canny complete.")

    def _handle_result(self, result: dict):
        if self._ui is None or self._image_view is None:
            return

        self._results = result
        if self._current_image is not None:
            self._results["original"] = self._current_image
            self._results["gray"] = self.to_grayscale_array(self._current_image)

        self._ui.cannyRunBtn.setEnabled(True)
        self._ui.cannySaveBtn.setEnabled(True)
        edges_index = next(
            index for index, (_, key) in enumerate(self.STAGES) if key == "edges"
        )
        self._ui.cannyStageCombo.setCurrentIndex(edges_index)

        edges = result.get("edges")
        if edges is not None:
            edge_pixels = int((edges > 0).sum())
            total_pixels = edges.size
            self._ui.cannyStatsLbl.setText(
                f"Size: {edges.shape[1]} × {edges.shape[0]}\n"
                f"Edge pixels: {edge_pixels:,}\n"
                f"Edge density: {edge_pixels / total_pixels * 100:.2f}%\n"
                f"Low threshold:  {result['low_thresh']:.1f}\n"
                f"High threshold: {result['high_thresh']:.1f}"
            )


"""Main/shared controller utilities and app wiring."""

from __future__ import annotations

from pathlib import Path

from PyQt5 import uic
from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot
from PyQt5.QtWidgets import QFileDialog, QMainWindow, QMessageBox, QVBoxLayout

from core.image_manager import load_image, save_image, to_grayscale, to_qpixmap


class MainController(QObject):
    """Base controller for common behavior across tab controllers."""

    result_ready = pyqtSignal(dict)
    status_message = pyqtSignal(str)
    error_occurred = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._worker = None
        self._last_result: dict | None = None

    def pick_image_path(self, parent) -> str:
        path, _ = QFileDialog.getOpenFileName(
            parent,
            "Open Image",
            str(Path.home()),
            "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)",
        )
        return path

    def load_image_array(self, path: str):
        return load_image(path)

    def save_image_array(self, array, path: str):
        save_image(array, path)

    def to_grayscale_array(self, image):
        return to_grayscale(image)

    def to_qpixmap_image(self, array):
        return to_qpixmap(array)

    def _start_worker(self, worker, running_message: str) -> bool:
        if self._worker and self._worker.isRunning():
            self.status_message.emit("Already processing. Please wait...")
            return False

        self.status_message.emit(running_message)
        self._worker = worker
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._worker.start()
        return True

    @pyqtSlot(dict)
    def _on_finished(self, result: dict):
        self._last_result = result
        self.result_ready.emit(result)

    @pyqtSlot(str)
    def _on_error(self, msg: str):
        self.error_occurred.emit(msg)

    @property
    def last_result(self) -> dict | None:
        return self._last_result


class AppController(QObject):
    """Top-level app controller that loads main.ui and wires tabs/controllers."""

    def __init__(self, window: QMainWindow):
        super().__init__(window)
        self.window = window
        uic.loadUi(str(Path(__file__).resolve().parents[1] / "ui" / "main.ui"), self.window)

        # Local imports avoid circular imports with MainController base usage.
        from controllers.canny_controller import CannyController
        from controllers.hough_controller import HoughController
        from controllers.snake_controller import SnakeController
        from controllers.contour_controller import ContourController

        self.canny_controller = CannyController(self.window)
        self.canny_controller.bind_ui(self.window)

        self.hough_controller = HoughController(self.window)
        self.hough_controller.bind_ui(self.window)

        self.snake_controller = SnakeController(self.window)
        self.snake_controller.bind_ui(self.window)

        self.contour_controller = ContourController(self.window)
        self.contour_controller.bind_ui(self.window)
        self.snake_controller.contours_changed.connect(
            self.contour_controller.on_contours_changed
        )

        self.window.actionOpenImage.triggered.connect(self._open_image_for_active_tab)
        self.window.actionQuit.triggered.connect(self.window.close)
        self.window.actionAbout.triggered.connect(self._about)

        self.canny_controller.status_message.connect(self.window.statusbar.showMessage)
        self.canny_controller.error_occurred.connect(self._show_error)
        self.hough_controller.status_message.connect(self.window.statusbar.showMessage)
        self.hough_controller.error_occurred.connect(self._show_error)
        self.snake_controller.status_message.connect(self.window.statusbar.showMessage)
        self.snake_controller.error_occurred.connect(self._show_error)
        self.contour_controller.status_message.connect(self.window.statusbar.showMessage)
        self.contour_controller.error_occurred.connect(self._show_error)

    def _open_image_for_active_tab(self):
        path, _ = QFileDialog.getOpenFileName(
            self.window,
            "Open Image",
            str(Path.home()),
            "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)",
        )
        if not path:
            return

        idx = self.window.tabWidget.currentIndex()
        if idx == 0:
            self.canny_controller.load_image(path)
        elif idx == 1:
            self.hough_controller.load_image(path)
        elif idx == 2:
            self.snake_controller.load_image(path)
        else:
            self.window.statusbar.showMessage("Open Image is available on Canny, Shapes, and Snake tabs.")

    def _show_error(self, msg: str):
        QMessageBox.critical(self.window, "Error", msg)

    def _about(self):
        QMessageBox.about(
            self.window,
            "About",
            "<b>Edge &amp; Contour Lab</b><br>"
            "Computer-Vision Assignment 2<br><br>"
            "<b>Member 1:</b> Canny Edge Detector (from scratch)<br>"
            "<b>Member 2:</b> Shape Detection (Hough)<br>"
            "<b>Member 3:</b> Active Contour (Snake)<br>"
            "<b>Member 4:</b> Chain Code &amp; Measurements",
        )
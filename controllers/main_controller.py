"""Main/shared controller utilities and app wiring."""

from __future__ import annotations

from pathlib import Path

from PyQt5 import uic
from PyQt5.QtCore import QObject
from PyQt5.QtWidgets import QFileDialog, QMainWindow, QMessageBox

from controllers.matching_controller import MatchingImagesController
from controllers.harris_controller import HarrisController


class AppController(QObject):
    """Top-level app controller that loads main.ui and wires tab controllers."""

    def __init__(self, window: QMainWindow):
        super().__init__(window)
        self.window = window

        # Load the UI from the .ui file
        ui_path = Path(__file__).resolve().parents[1] / "ui" / "main.ui"
        uic.loadUi(str(ui_path), self.window)

        # ── Tab controllers ──────────────────────────────────────────────
        self.matching_controller = MatchingImagesController(self.window)
        self.matching_controller.bind_ui(self.window)

        self.harris_controller = HarrisController(self.window)
        self.harris_controller.bind_ui(self.window)

        # ── Menu actions ─────────────────────────────────────────────────
        self.window.actionOpenImage.triggered.connect(self._open_image_for_active_tab)
        self.window.actionQuit.triggered.connect(self.window.close)
        self.window.actionAbout.triggered.connect(self._about)

        # ── Status / error signals ───────────────────────────────────────
        self.matching_controller.status_message.connect(self.window.statusbar.showMessage)
        self.matching_controller.error_occurred.connect(self._show_error)

        self.harris_controller.status_message.connect(self.window.statusbar.showMessage)
        self.harris_controller.error_occurred.connect(self._show_error)

    def _open_image_for_active_tab(self):
        """Open an image and delegate it to the controller of the currently active tab."""
        QMessageBox.information(
            self.window,
            "Information",
            "Please use the 'Load Image' button on the active tab.",
        )

    def _show_error(self, msg: str):
        """Display a critical error message."""
        QMessageBox.critical(self.window, "Error", msg)

    def _about(self):
        """Show the about dialog."""
        QMessageBox.about(
            self.window,
            "About",
            "<b>Feature Extraction & Matching Lab</b><br><br>"
            "Demonstrates Harris and λ- corner detection, "
            "SIFT feature detection and descriptor matching.<br>"
            "Uses SSD and NCC for descriptor matching.",
        )
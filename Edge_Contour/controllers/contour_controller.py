"""
contour_controller.py — Chain Code & Measurements for the Snake tab.
"""

from __future__ import annotations

import numpy as np
from pathlib import Path

from PyQt5.QtCore import pyqtSlot
from PyQt5.QtWidgets import QFileDialog

from controllers.main_controller import MainController
from core.contour.chain_code import (
    compute_chain_code,
    chain_code_to_string,
    chain_code_difference,
)
from core.contour.perimeter import compute_perimeter
from core.contour.area import compute_area


class ContourController(MainController):
    """Handles chain code computation and display inside the Snake tab."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._ui = None
        self._contour: np.ndarray | None = None
        self._chain_code: list[int] = []
        self._perimeter: float = 0.0
        self._area: float = 0.0

    # ── Public API ──────────────────────────────────────────────────────────────

    def bind_ui(self, ui):
        self._ui = ui
        ui.contourExportBtn.clicked.connect(self._export)
        self._clear_display()

    @pyqtSlot(object, object)
    def on_contours_changed(self, initial: np.ndarray | None, final: np.ndarray | None):
        self._contour = final
        if final is not None and len(final) >= 3:
            self._compute_and_display(final)
        else:
            self._clear_display()

    # ── Internal ────────────────────────────────────────────────────────────────

    def _compute_and_display(self, contour: np.ndarray):
        self._chain_code = compute_chain_code(contour)
        self._perimeter  = compute_perimeter(self._chain_code)
        self._area       = compute_area(contour)

        code_str = chain_code_to_string(self._chain_code)
        diff_str = chain_code_to_string(chain_code_difference(self._chain_code))

        ui = self._ui

        # Left panel — measurements
        ui.contourPointsVal.setText(str(len(contour)))
        ui.contourPerimeterVal.setText(f"{self._perimeter:.2f} px")
        ui.contourAreaVal.setText(f"{self._area:.2f} px²")

        # Below canvas — chain code labels (word-wrap handles long strings)
        ui.contourCodeVal.setText(code_str)
        ui.contourDiffVal.setText(diff_str)

        ui.contourExportBtn.setEnabled(True)
        self.status_message.emit(
            f"Chain code: {len(self._chain_code)} codes  |  "
            f"Perimeter: {self._perimeter:.1f} px  |  "
            f"Area: {self._area:.1f} px²"
        )

    def _clear_display(self):
        if self._ui is None:
            return
        for w in (
                self._ui.contourPointsVal,
                self._ui.contourPerimeterVal,
                self._ui.contourAreaVal,
                self._ui.contourCodeVal,
                self._ui.contourDiffVal,
        ):
            w.setText("—")
        self._ui.contourExportBtn.setEnabled(False)

    def _export(self):
        if not self._chain_code:
            return
        path, _ = QFileDialog.getSaveFileName(
            None,
            "Save Chain Code & Measurements",
            str(Path.home() / "chain_code.txt"),
            "Text (*.txt)",
        )
        if not path:
            return

        diff = chain_code_difference(self._chain_code)
        content = (
                "Chain Code & Measurements\n"
                + "=" * 40 + "\n"
                + f"Points     : {len(self._contour)}\n"
                + f"Perimeter  : {self._perimeter:.4f} px\n"
                + f"Area       : {self._area:.4f} px²\n\n"
                + f"Chain code ({len(self._chain_code)} codes):\n"
                + f"{chain_code_to_string(self._chain_code)}\n\n"
                + "First-difference (rotation-invariant):\n"
                + f"{chain_code_to_string(diff)}\n"
        )
        Path(path).write_text(content)
        self.status_message.emit(f"Saved to {Path(path).name}")

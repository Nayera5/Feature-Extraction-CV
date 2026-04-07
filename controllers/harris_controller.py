"""
Controller for the "Feature Detection" tab.

Handles:
  • Image loading (via button or drag-and-drop placeholder)
  • Running Harris or λ- detection with user-specified parameters
  • Displaying the result image in canvasLabel (QLabel)
  • Showing computation stats in statsLabel
"""

from __future__ import annotations

import traceback
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PyQt5.QtCore import QObject, Qt, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QFileDialog, QLabel, QMainWindow

from detectors.harris_detector import detect_harris
from detectors.lambda_detector import detect_lambda


class HarrisController(QObject):
    """Wires all widgets on the Feature Detection tab."""

    status_message = pyqtSignal(str)
    error_occurred = pyqtSignal(str)

    # ── Widget names expected in main.ui ─────────────────────────────────
    # Buttons
    _BTN_LOAD   = "harrisBtnLoad"
    _BTN_RUN    = "harrisBtnRun"
    _BTN_SAVE   = "harrisBtnSave"

    # Display
    _LBL_CANVAS = "harrisCanvasLabel"   # QLabel where result is painted
    _LBL_NAME   = "harrisImageNameLbl"  # shows loaded filename
    _LBL_STATS  = "harrisStatsLbl"      # computation time / corner count

    # Parameters
    _CMB_METHOD     = "harrisCmbMethod"       # QComboBox: "Harris" / "λ-"
    _SPN_THRESHOLD  = "harrisSpnThreshold"    # QDoubleSpinBox  (0.001 – 1.0)
    _SPN_K          = "harrisSpnK"            # QDoubleSpinBox  (harris k, 0.01-0.2)
    _SPN_SIGMA      = "harrisSpnSigma"        # QDoubleSpinBox  (gaussian sigma)
    _SPN_KSIZE      = "harrisSpnKsize"        # QSpinBox        (gaussian ksize)
    _SPN_MINDIST    = "harrisSpnMinDist"      # QSpinBox        (NMS min distance)
    _SPN_MAXCORNERS = "harrisSpnMaxCorners"   # QSpinBox        (max corners)

    def __init__(self, window: QMainWindow):
        super().__init__(window)
        self.window = window
        self._image: Optional[np.ndarray] = None        # original loaded image
        self._result_vis: Optional[np.ndarray] = None   # last result BGR image

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def bind_ui(self, window: QMainWindow) -> None:
        """Connect all widget signals.  Call once after uic.loadUi()."""
        self._w = window  # shorthand

        btn_load = self._widget(_BTN := self._BTN_LOAD)
        btn_run  = self._widget(self._BTN_RUN)
        btn_save = self._widget(self._BTN_SAVE)

        if btn_load:
            btn_load.clicked.connect(self._on_load)
        if btn_run:
            btn_run.clicked.connect(self._on_run)
            btn_run.setEnabled(False)
        if btn_save:
            btn_save.clicked.connect(self._on_save)
            btn_save.setEnabled(False)

        # populate combo if present
        cmb = self._widget(self._CMB_METHOD)
        if cmb:
            cmb.clear()
            cmb.addItems(["Harris", "λ-  (Shi-Tomasi)"])

        # set sensible defaults on spinboxes
        self._set_spinbox_defaults()

        # canvas placeholder text
        canvas = self._widget(self._LBL_CANVAS)
        if canvas:
            canvas.setAlignment(Qt.AlignCenter)
            canvas.setText("Load an image to begin")

    # ------------------------------------------------------------------
    # Private – slots
    # ------------------------------------------------------------------

    def _on_load(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self.window,
            "Open Image",
            "",
            "Images (*.png *.jpg *.jpeg *.bmp *.tiff *.tif *.webp)",
        )
        if not path:
            return

        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            self.error_occurred.emit(f"Failed to load image:\n{path}")
            return

        self._image = img
        self._result_vis = None

        # update name label
        lbl_name = self._widget(self._LBL_NAME)
        if lbl_name:
            lbl_name.setText(Path(path).name)

        # show original on canvas
        self._show_image_on_canvas(img)

        # enable run button
        btn_run = self._widget(self._BTN_RUN)
        if btn_run:
            btn_run.setEnabled(True)

        # reset stats
        lbl_stats = self._widget(self._LBL_STATS)
        if lbl_stats:
            lbl_stats.setText("Image loaded — press Run to detect corners.")

        self.status_message.emit(f"Loaded: {Path(path).name}")

    def _on_run(self) -> None:
        if self._image is None:
            return

        # ── Read parameters ─────────────────────────────────────────────
        method    = self._combo_text(self._CMB_METHOD)
        threshold = self._spin_value(self._SPN_THRESHOLD, 0.01)
        k_harris  = self._spin_value(self._SPN_K, 0.04)
        sigma     = self._spin_value(self._SPN_SIGMA, 1.0)
        ksize     = int(self._spin_value(self._SPN_KSIZE, 5))
        min_dist  = int(self._spin_value(self._SPN_MINDIST, 5))
        max_corn  = int(self._spin_value(self._SPN_MAXCORNERS, 2000))

        # ksize must be odd
        if ksize % 2 == 0:
            ksize += 1

        try:
            if method.startswith("Harris"):
                result = detect_harris(
                    self._image,
                    k=k_harris,
                    threshold_rel=threshold,
                    gaussian_ksize=ksize,
                    gaussian_sigma=sigma,
                    min_dist=min_dist,
                    max_corners=max_corn,
                )
                method_label = "Harris"
                extra = f"k = {k_harris:.4f}"
            else:
                result = detect_lambda(
                    self._image,
                    threshold_rel=threshold,
                    gaussian_ksize=ksize,
                    gaussian_sigma=sigma,
                    min_dist=min_dist,
                    max_corners=max_corn,
                )
                method_label = "λ-  (Shi-Tomasi)"
                extra = ""

        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            self.error_occurred.emit(f"Detection failed:\n{exc}")
            return

        self._result_vis = result.visualisation

        # ── Update canvas ────────────────────────────────────────────────
        self._show_image_on_canvas(result.visualisation)

        # ── Update stats label ───────────────────────────────────────────
        stats_lines = [
            f"Method : {method_label}",
            f"Corners : {result.num_corners}",
            f"Time : {result.computation_time_ms:.2f} ms",
            f"Threshold : {threshold:.4f}",
            f"Gauss : σ={sigma}  k={ksize}×{ksize}",
            f"NMS dist : {min_dist} px",
        ]
        if extra:
            stats_lines.insert(2, extra)

        lbl_stats = self._widget(self._LBL_STATS)
        if lbl_stats:
            lbl_stats.setText("\n".join(stats_lines))

        # enable save
        btn_save = self._widget(self._BTN_SAVE)
        if btn_save:
            btn_save.setEnabled(True)

        self.status_message.emit(
            f"{method_label}: {result.num_corners} corners in "
            f"{result.computation_time_ms:.1f} ms"
        )

    def _on_save(self) -> None:
        if self._result_vis is None:
            return

        path, _ = QFileDialog.getSaveFileName(
            self.window,
            "Save Result",
            "corners_result.png",
            "PNG (*.png);;JPEG (*.jpg *.jpeg);;BMP (*.bmp)",
        )
        if not path:
            return

        ok = cv2.imwrite(path, self._result_vis)
        if ok:
            self.status_message.emit(f"Saved → {Path(path).name}")
        else:
            self.error_occurred.emit(f"Could not save image to:\n{path}")

    # ------------------------------------------------------------------
    # Private – helpers
    # ------------------------------------------------------------------

    def _widget(self, name: str):
        """Find a widget by object name; return None if not found."""
        return self.window.findChild(object, name)  # type: ignore[arg-type]

    def _combo_text(self, name: str, default: str = "") -> str:
        w = self._widget(name)
        return w.currentText() if w else default

    def _spin_value(self, name: str, default: float = 0.0) -> float:
        w = self._widget(name)
        return w.value() if w else default

    def _set_spinbox_defaults(self) -> None:
        """Set sensible initial values on spinboxes (safe if widgets absent)."""
        defs = {
            self._SPN_THRESHOLD:  (0.01,  0.001,  1.0,  3),
            self._SPN_K:          (0.04,  0.01,   0.2,  3),
            self._SPN_SIGMA:      (1.0,   0.1,    5.0,  1),
            self._SPN_KSIZE:      (5,     3,      21,   0),   # int
            self._SPN_MINDIST:    (5,     1,      50,   0),   # int
            self._SPN_MAXCORNERS: (2000,  100,    10000, 0),  # int
        }
        for name, (val, mn, mx, dec) in defs.items():
            w = self._widget(name)
            if w is None:
                continue
            w.setMinimum(mn)
            w.setMaximum(mx)
            if dec > 0:
                w.setDecimals(dec)
            w.setValue(val)

    def _show_image_on_canvas(self, img_bgr: np.ndarray) -> None:
        """Convert a BGR numpy array to QPixmap and show it in the canvas label."""
        canvas: Optional[QLabel] = self._widget(self._LBL_CANVAS)
        if canvas is None:
            return

        h, w = img_bgr.shape[:2]
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        qimg = QImage(img_rgb.data, w, h, 3 * w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)

        # Scale to fit the label while keeping aspect ratio
        canvas.setPixmap(
            pixmap.scaled(
                canvas.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
        )
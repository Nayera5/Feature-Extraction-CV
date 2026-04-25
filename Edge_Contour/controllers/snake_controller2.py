from pathlib import Path

import numpy as np

from PyQt5.QtCore import QObject, QEvent, QPoint, Qt, QThread, pyqtSignal
from PyQt5.QtGui import QBrush, QPainter, QPen, QPixmap, QColor, QPolygon
from PyQt5.QtWidgets import QLabel

from controllers.main_controller import MainController
from core.snake.greedy_algorithm import evolve_snake


class _SnakeCanvasInteractor(QObject):
	"""Attach contour-drawing and overlay rendering to a QLabel from main.ui."""

	contour_drawn = pyqtSignal(object)

	def __init__(self, label: QLabel, pixmap_factory, parent=None):
		super().__init__(parent)
		self._label = label
		self._pixmap_factory = pixmap_factory

		self._base_pixmap = None
		self._image_shape = None
		self._points = []
		self._initial = None
		self._final = None

		self._label.installEventFilter(self)
		self._label.setAlignment(Qt.AlignCenter)
		self._label.setMouseTracking(True)

	def set_image(self, image: np.ndarray):
		self._base_pixmap = self._pixmap_factory(image)
		self._image_shape = image.shape[:2]
		self._points = []
		self._initial = None
		self._final = None
		self._refresh()

	def set_contours(self, initial, final):
		self._initial = initial
		self._final = final
		self._refresh()

	def eventFilter(self, obj, event):
		if obj is not self._label:
			return False

		if event.type() == QEvent.Resize:
			self._refresh()
			return False

		if getattr(self, "_base_pixmap", None) is None:
			return False

		if event.type() == QEvent.MouseButtonPress and event.button() == Qt.LeftButton:
			p = self._widget_to_image(event.pos())
			if p is not None:
				self._points.append(p)
				self._refresh(preview=True)
			return True

		if event.type() == QEvent.MouseButtonPress and event.button() == Qt.RightButton:
			if len(self._points) >= 3:
				contour = np.array(self._points, dtype=int)
				self._points = []
				self.contour_drawn.emit(contour)
				self._refresh(preview=False)
			return True

		return False

	def _display_geometry(self):
		if self._image_shape is None:
			return None
		h, w = self._image_shape
		if h <= 0 or w <= 0:
			return None
		scale = min(self._label.width() / w, self._label.height() / h)
		disp_w = w * scale
		disp_h = h * scale
		off_x = (self._label.width() - disp_w) / 2
		off_y = (self._label.height() - disp_h) / 2
		return off_x, off_y, scale, w, h

	def _widget_to_image(self, pos: QPoint):
		geom = self._display_geometry()
		if geom is None:
			return None
		off_x, off_y, scale, w, h = geom
		x = int(max(0, min(w - 1, (pos.x() - off_x) / scale)))
		y = int(max(0, min(h - 1, (pos.y() - off_y) / scale)))
		return x, y

	def _image_to_display(self, x: int, y: int):
		geom = self._display_geometry()
		if geom is None:
			return None
		off_x, off_y, scale, _, _ = geom
		return int(round(off_x + x * scale)), int(round(off_y + y * scale))

	def _refresh(self, preview: bool = False):
		if getattr(self, "_base_pixmap", None) is None:
			return

		pix = self._base_pixmap.scaled(
			self._label.size(),
			Qt.KeepAspectRatio,
			Qt.SmoothTransformation,
		)

		# Draw overlays directly on a display-size canvas aligned to QLabel.
		canvas = QPixmap(self._label.size())
		canvas.fill(QColor("#1e1e2e"))
		painter = QPainter(canvas)
		painter.drawPixmap((self._label.width() - pix.width()) // 2, (self._label.height() - pix.height()) // 2, pix)

		if preview and len(self._points) > 0:
			painter.setPen(QPen(QColor(255, 220, 50), 2))
			painter.setBrush(QBrush(QColor(255, 220, 50, 100)))
			for x, y in self._points:
				p = self._image_to_display(int(x), int(y))
				if p is not None:
					painter.drawEllipse(QPoint(p[0], p[1]), 3, 3)
			if len(self._points) > 1:
				for i in range(len(self._points) - 1):
					p1 = self._image_to_display(int(self._points[i][0]), int(self._points[i][1]))
					p2 = self._image_to_display(int(self._points[i+1][0]), int(self._points[i+1][1]))
					if p1 and p2:
						painter.drawLine(p1[0], p1[1], p2[0], p2[1])

		if self._initial is not None and len(self._initial) > 1:
			painter.setPen(QPen(Qt.green, 2))
			self._draw_closed_poly(painter, self._initial)

		if self._final is not None and len(self._final) > 1:
			painter.setBrush(QBrush(QColor(80, 64, 200, 90)))
			painter.setPen(QPen(Qt.blue, 2))
			self._draw_closed_poly(painter, self._final)

		painter.end()
		self._label.setPixmap(canvas)
		self._label.setText("")

	def _draw_closed_poly(self, painter: QPainter, contour):
		points = []
		for x, y in contour:
			p = self._image_to_display(int(x), int(y))
			if p is not None:
				points.append(QPoint(p[0], p[1]))
		if len(points) < 2:
			return

		polygon = QPolygon(points)
		painter.drawPolygon(polygon)


class _SnakeWorker(QThread):
    """Runs snake evolution in a background thread."""

    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, image: np.ndarray, contour: np.ndarray, params: dict, parent=None):
        super().__init__(parent)
        self._image = image
        self._contour = contour
        self._params = params

    def run(self):
        try:
            result = evolve_snake(
                self._image,
                self._contour,
                num_iterations=self._params.get("num_iterations", 50),
                alpha=self._params.get("alpha", 0.1),
                beta=self._params.get("beta", 0.1),
            )
            self.finished.emit(result)
        except Exception as exc:
            self.error.emit(str(exc))


class SnakeController(MainController):
	contours_changed = pyqtSignal(object, object)  # initial, final

	def __init__(self, parent=None):
		super().__init__(parent)
		self._image: np.ndarray | None = None
		self._initial_contour: np.ndarray | None = None
		self._final_contour: np.ndarray | None = None
		self._ui = None
		self._canvas: _SnakeCanvasInteractor | None = None

	def bind_ui(self, ui):
		self._ui = ui
		self._canvas = _SnakeCanvasInteractor(ui.snakeCanvasHost, self.to_qpixmap_image, parent=self)
		self._canvas.contour_drawn.connect(self._on_contour_drawn)
		self.contours_changed.connect(self._canvas.set_contours)

		ui.snakeOpenBtn.clicked.connect(self.open_image_dialog)
		ui.snakeRunBtn.clicked.connect(self.run_from_ui)
		ui.snakeClearBtn.clicked.connect(self.clear_initial_contour)

	def open_image_dialog(self):
		if self._ui is None:
			return
		path = self.pick_image_path(self._ui)
		if path:
			self.load_image(path)

	def load_image(self, path: str):
		if self._ui is None or self._canvas is None:
			return
		try:
			image = self.load_image_array(path)
		except Exception as exc:  # noqa: BLE001
			self.error_occurred.emit(str(exc))
			return

		self._image = image
		self._initial_contour = None
		self._final_contour = None
		self._canvas.set_image(image)
		self._ui.snakeImageNameLbl.setText(Path(path).name)
		self._ui.snakeRunBtn.setEnabled(False)
		self._ui.snakeClearBtn.setEnabled(False)
		self._ui.snakeStatusLbl.setText("Left-click to draw contour, right-click to close")
		self.contours_changed.emit(None, None)

	def _on_contour_drawn(self, contour: np.ndarray):
		if self._image is None or self._ui is None:
			return
		self._initial_contour = self._resample_contour(
			contour,
			self._ui.snakePointsSpin.value(),
		)
		self._final_contour = None
		self.contours_changed.emit(self._initial_contour, None)
		self._ui.snakeRunBtn.setEnabled(True)
		self._ui.snakeClearBtn.setEnabled(True)
		self._ui.snakeStatusLbl.setText(f"Contour ready with {len(self._initial_contour)} points")

	def clear_initial_contour(self):
		self._initial_contour = None
		self._final_contour = None
		self.contours_changed.emit(None, None)
		if self._ui is not None:
			self._ui.snakeRunBtn.setEnabled(False)
			self._ui.snakeClearBtn.setEnabled(False)
			self._ui.snakeStatusLbl.setText("Contour cleared")

	def run_from_ui(self):
		if self._ui is None:
			return
		params = {
			"num_iterations": self._ui.snakeItersSpin.value(),
			"alpha": self._ui.snakeAlphaSpin.value(),
			"beta": self._ui.snakeBetaSpin.value(),
		}
		self._ui.snakeRunBtn.setEnabled(False)
		self.run(params)

	def run(self, params: dict):
		if self._image is None or self._initial_contour is None:
			self.error_occurred.emit("Load image and draw a contour first.")
			return

		worker = _SnakeWorker(self._image, self._initial_contour.copy(), params, parent=self)
		self._start_worker(worker, "Running greedy snake evolution...")

	def _on_finished(self, result: dict):
		super()._on_finished(result)
		self._final_contour = result.get("contour")
		self.contours_changed.emit(self._initial_contour, self._final_contour)
		self.status_message.emit("Snake evolution complete.")
		if self._ui is not None:
			self._ui.snakeRunBtn.setEnabled(True)
			if self._initial_contour is None or self._final_contour is None:
				self._ui.snakeStatusLbl.setText("No contour result")
			else:
				shift = np.linalg.norm(self._final_contour - self._initial_contour, axis=1)
				self._ui.snakeStatusLbl.setText(f"Done. mean shift={float(shift.mean()):.2f}px")

	@staticmethod
	def _resample_contour(contour: np.ndarray, target_points: int) -> np.ndarray:
		"""Resample a closed contour to target number of points."""
		if contour.shape[0] < 3:
			return contour.astype(int)

		closed = np.vstack([contour, contour[0]])
		diff = np.diff(closed, axis=0)
		segs = np.sqrt((diff ** 2).sum(axis=1))
		perim = segs.sum()
		if perim < 1e-8:
			return contour.astype(int)

		cum = np.concatenate([[0.0], np.cumsum(segs)])
		target_dists = np.linspace(0, perim, target_points, endpoint=False)
		resampled = np.zeros((target_points, 2))

		for i, s in enumerate(target_dists):
			idx = int(np.searchsorted(cum, s, side="right") - 1)
			idx = min(idx, len(segs) - 1)
			seg_len = segs[idx]
			if seg_len < 1e-8:
				resampled[i] = closed[idx]
			else:
				t = (s - cum[idx]) / seg_len
				resampled[i] = closed[idx] + t * (closed[idx + 1] - closed[idx])

		return np.rint(resampled).astype(int)



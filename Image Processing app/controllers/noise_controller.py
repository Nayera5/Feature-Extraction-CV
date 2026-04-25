import cv2
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt


class NoiseController:
    def __init__(self, window, image_manager):
        self.ui = window
        self.image_manager = image_manager
        self.noisy_image = None

        self._setup_ui()
        self._connect_signals()

        # Display original image in input group at start
        self.display_input_image()

    def _setup_ui(self):
        # Noise types
        self.ui.noise_combo_type.addItems(
            ["Gaussian", "Uniform", "Salt & Pepper"]
        )

        # Filters
        self.ui.noise_combo_filter.addItems(
            ["Average (3x3)", "Gaussian (3x3)", "Median (3x3)"]
        )

    def _connect_signals(self):
        # Buttons
        self.ui.noise_btn_apply.clicked.connect(self.apply_noise)
        self.ui.filter_btn_apply.clicked.connect(self.apply_filter)

        # Auto-apply noise when slider changes
        self.ui.noise_slider_amount.valueChanged.connect(self.apply_noise)

    def display_input_image(self):
        """Always display the original image in its group box"""
        image = self.image_manager.original_image
        if image is not None:
            self.display_image(image, self.ui.noise_input_image)

    def apply_noise(self):
        image = self.image_manager.original_image
        if image is None:
            return

        # Display original image (keeps input in place)
        self.display_input_image()

        # Add noise
        noise_type = self.ui.noise_combo_type.currentText()
        amount = self.ui.noise_slider_amount.value() / 100.0

        from core.noise import add_noise

        self.noisy_image = add_noise(image, noise_type, amount)

        # Display noisy image, expand to fill its group box
        self.display_image(self.noisy_image, self.ui.noise_noisy_image)

    def apply_filter(self):
        image = self.noisy_image
        if image is None:
            return

        filter_type = self.ui.noise_combo_filter.currentText()

        from core.filters import apply_filter

        filtered_image = apply_filter(image, filter_type)

        # Display filtered image, expand to fill its group box
        self.display_image(filtered_image, self.ui.noise_filtered_image)

    def display_image(self, image, label, expand=True):
        """Display image in QLabel.
        expand=True → scale pixmap to label size
        expand=False → show original size
        """
        if image is None:
            return

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)

        if expand:
            # Scale pixmap to label size while keeping aspect ratio
            pixmap = pixmap.scaled(
                label.width(),
                label.height(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )

        label.setPixmap(pixmap)
        label.setAlignment(Qt.AlignCenter)
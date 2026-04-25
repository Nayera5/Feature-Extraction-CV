from PyQt5.QtWidgets import QFileDialog
from PyQt5.QtCore import Qt, QEvent, QObject
import cv2
from core.hybrid import create_hybrid_image
from controllers.main_controller import MainController


class HybridController(QObject):
    def __init__(self, window):
        super().__init__()
        self.window = window

        self.image1 = None
        self.image2 = None

        # ── Setup labels using MainController's helper ──
        MainController._setup_label(self, self.window.hybrid_label_img1, dashed=True, clickable=True)
        MainController._setup_label(self, self.window.hybrid_label_img2, dashed=True, clickable=True)
        MainController._setup_label(self, self.window.hybrid_label_result)

        # ── Install event filters for double-click ──
        self.window.hybrid_label_img1.installEventFilter(self)
        self.window.hybrid_label_img2.installEventFilter(self)

        # ── Populate combo boxes ──
        self.window.hybrid_combo_filter1.addItems(["Low Pass", "High Pass"])
        self.window.hybrid_combo_filter2.addItems(["Low Pass", "High Pass"])
        self.window.hybrid_combo_filter2.setCurrentIndex(1)  # default image2 → High Pass

        # ── Sliders → update value labels ──
        self.window.hybrid_slider_cutoff1.valueChanged.connect(
            lambda v: self.window.hybrid_label_cutoff1_val.setText(str(v))
        )
        self.window.hybrid_slider_cutoff2.valueChanged.connect(
            lambda v: self.window.hybrid_label_cutoff2_val.setText(str(v))
        )
        self.window.hybrid_slider_alpha.valueChanged.connect(
            lambda v: self.window.hybrid_label_alpha_val.setText(f"{v / 100:.2f}")
        )

        # ── Button ──
        self.window.hybrid_btn_create.clicked.connect(self.create_hybrid)

    # ──────────────────────────────────────────────
    #  Double-click to load images
    # ──────────────────────────────────────────────
    def eventFilter(self, obj, event):
        if event.type() == QEvent.MouseButtonDblClick and event.button() == Qt.LeftButton:
            if obj == self.window.hybrid_label_img1:
                self.load_image(1); return True
            if obj == self.window.hybrid_label_img2:
                self.load_image(2); return True
        return False

    def load_image(self, slot: int):
        path, _ = QFileDialog.getOpenFileName(
            self.window, "Select Image", "", "Images (*.png *.jpg *.bmp *.jpeg)"
        )
        if not path:
            return
        img = cv2.imread(path)
        if img is None:
            return
        if slot == 1:
            self.image1 = img
            MainController.display_image(self, img, self.window.hybrid_label_img1)
        else:
            self.image2 = img
            MainController.display_image(self, img, self.window.hybrid_label_img2)

    # ──────────────────────────────────────────────
    #  Create hybrid and show result
    # ──────────────────────────────────────────────
    def create_hybrid(self):
        if self.image1 is None or self.image2 is None:
            return

        cutoff1   = self.window.hybrid_slider_cutoff1.value()
        low_pass1 = self.window.hybrid_combo_filter1.currentText() == "Low Pass"

        cutoff2   = self.window.hybrid_slider_cutoff2.value()
        low_pass2 = self.window.hybrid_combo_filter2.currentText() == "Low Pass"

        alpha = self.window.hybrid_slider_alpha.value() / 100.0

        hybrid, _, _ = create_hybrid_image(
            self.image1,
            self.image2,
            low_cutoff=cutoff1,
            high_cutoff=cutoff2,
            alpha=alpha,
            low_pass1=low_pass1,
            low_pass2=low_pass2,
        )

        MainController.display_image(self, hybrid, self.window.hybrid_label_result)
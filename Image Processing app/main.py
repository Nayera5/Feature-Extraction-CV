import sys
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5 import uic
from controllers.main_controller import MainController
from controllers.hybrid_controller import HybridController
from controllers.noise_controller import NoiseController


if __name__ == "__main__":
    app = QApplication(sys.argv)

    window = QMainWindow()
    uic.loadUi("ui/dark.ui", window)

    controller = MainController(window)
    hybrid_controller = HybridController(window)
    noise_controller = NoiseController(window, controller.manager)

    # Set default tab to Input tab (index 0)
    window.tabWidget.setCurrentIndex(0)

    window.show()
    sys.exit(app.exec_())
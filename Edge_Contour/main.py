"""
main.py — Application entry point.

Launches the edge-contour-lab PyQt5 window.
"""

import sys
from pathlib import Path

from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.QtCore    import Qt

from controllers.main_controller import AppController


def main():
    # High-DPI support
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps,    True)

    app = QApplication(sys.argv)
    app.setApplicationName("Edge & Contour Lab")
    app.setOrganizationName("CV-Group")

    # Load stylesheet
    qss_path = Path(__file__).parent / "ui" / "dark.qss"
    if qss_path.exists():
        app.setStyleSheet(qss_path.read_text())

    window = QMainWindow()
    app_controller = AppController(window)
    window._app_controller = app_controller
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
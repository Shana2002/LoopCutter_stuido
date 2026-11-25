from __future__ import annotations

import sys

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication

from autocutter.ui.main_window import MainWindow


def main() -> int:
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    app = QApplication(sys.argv)
    app.setApplicationName("AutoCutter Studio")
    window = MainWindow()
    window.show()
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())

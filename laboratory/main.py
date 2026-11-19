import os
import sys

# Repo root first on sys.path so `highspl` and `laboratory.*` resolve without
# depending on the working directory.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from laboratory.ui.main_window import MainWindow, QApplication

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

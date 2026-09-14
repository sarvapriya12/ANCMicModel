import os
import sys

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from ui.main_window import MainWindow, QApplication

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

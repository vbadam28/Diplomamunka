import sys

from PySide6.QtWidgets import QApplication
from gui.window import MainWindow


if __name__ == '__main__':
    app = QApplication(sys.argv)
    wd = MainWindow()
    wd.show()
    app.exec()


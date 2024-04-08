#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/4/3 17:28
# @Author  : Hongfei Wang
# @File    : main.py
# @Software: PyCharm
from PyQt5.QtWidgets import *
import sys
from untitled import Ui_MainWindow


class Window(QMainWindow):
    def __init__(self):
        super().__init__()
        self.main_frame = Ui_MainWindow()
        self.main_frame.setupUi(self)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = Window()
    window.show()
    sys.exit(app.exec_())

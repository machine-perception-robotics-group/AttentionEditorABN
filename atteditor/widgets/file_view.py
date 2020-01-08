#!/usr/bin/env python
# -*- coding: utf-8 -*-

# file_view.py
# Copyright (c) Tsubasa Hirakawa, 2020


from os.path import splitext
import cv2
from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt

from ..config import IMAGE_DIR


IMAGE_FILE_FILTER = ['*.jpg', '*.png', '*.bmp']
IMAGE_FILE_EXTENSION = ['.jpg', '.png', '.bmp']


class FileView(QtWidgets.QWidget):

    def __init__(self):
        super(FileView, self).__init__()
        self.current_path = IMAGE_DIR

        # file system model
        self.model = QtWidgets.QFileSystemModel()
        self.model.setNameFilters(IMAGE_FILE_FILTER)

        # tree view
        self.tree = QtWidgets.QTreeView()
        self.tree.setModel(self.model)
        self.tree.setRootIndex(self.model.setRootPath(self.current_path))
        self.tree.setColumnHidden(1, True)
        self.tree.setColumnHidden(2, True)
        self.tree.setColumnHidden(3, True)
        self.tree.setAnimated(False)
        self.tree.setIndentation(20)
        self.tree.setSortingEnabled(True)
        self.tree.sortByColumn(0, Qt.AscendingOrder)

        # buttons
        self.button_change = QtWidgets.QPushButton('change directory')
        self.button_change.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        self.button_change.clicked.connect(self.change_directory)

        # layout
        _layout = QtWidgets.QVBoxLayout()
        _layout.addWidget(self.tree)
        _layout.addWidget(self.button_change)
        self.setLayout(_layout)

    def change_directory(self):
        _selected_path = QtWidgets.QFileDialog.getExistingDirectory(self,
                                                                    caption="Open Image Directory",
                                                                    directory=self.current_path)
        self.current_path = _selected_path

        if _selected_path != '':
            self.tree.setRootIndex(self.model.setRootPath(_selected_path))

    def select_file(self):
        _index = self.tree.currentIndex()
        path = self.model.filePath(_index)
        _, ext = splitext(path)
        if ext in IMAGE_FILE_EXTENSION:
            _image = cv2.imread(path)
            return _image
        else:
            return None


if __name__ == '__main__':
    import sys

    app = QtWidgets.QApplication(sys.argv)
    widget = FileView()
    widget.setGeometry(100, 100, 500, 500)
    widget.show()
    sys.exit(app.exec_())

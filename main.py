#!/usr/bin/env python
# -*- coding: utf-8 -*-

# main.py
# Copyright (c) Tsubasa Hirakawa, 2020


import sys

from PyQt5 import QtWidgets, QtCore
from atteditor.widgets import *
from atteditor.utils import *


class MainWindow(QtWidgets.QMainWindow):

    def __init__(self):
        super(MainWindow, self).__init__()

        # widgets ###########
        # image view  : widget for displaying image and attention map and for editing attention map
        # file view   : widget for selecting image data
        # network view: widget for managing network model and displaying classification results
        #####################

        # widgets
        self.image_view = ImageView()
        self.file_view = FileView()
        self.network_view = NetworkView()

        # widget dock layout
        self.setCentralWidget(self.image_view)

        self.file_view_dock = QtWidgets.QDockWidget("Image Files", self)
        self.file_view_dock.setWidget(self.file_view)
        self.addDockWidget(QtCore.Qt.LeftDockWidgetArea, self.file_view_dock)

        self.network_view_dock = QtWidgets.QDockWidget("Network Model", self)
        self.network_view_dock.setWidget(self.network_view)
        self.addDockWidget(QtCore.Qt.RightDockWidgetArea, self.network_view_dock)

        # functions
        self.file_view.tree.doubleClicked.connect(self.load_image)
        self.network_view.button_infer.clicked.connect(self.inference)

        self.set_menu_bar()

        menu_bar = self.menuBar()

    def load_image(self):
        _image_filename = self.file_view.select_file()
        if _image_filename is not None:
            _image = center_crop(_image_filename)
            self.image_view.reset_image(_image)

    def inference(self):
        if self.image_view.orig_image is not None:
            _att_map = self.network_view.infer(self.image_view.orig_image,
                                               input_att_map=self.image_view.get_attention_map())
            self.image_view.update_attention_image(_att_map)

    def set_menu_bar(self):
        menu_bar = self.menuBar()

        # file menu
        menu_file = menu_bar.addMenu("File")

        action_file_open = QtWidgets.QAction("Change Directory", self)
        action_file_open.setShortcut("Ctrl+O")
        action_file_open.triggered.connect(self.file_view.change_directory)
        menu_file.addAction(action_file_open)

        # menu_edit = menu_bar.addMenu("Edit")

        menu_view = menu_bar.addMenu("View")
        action_show_file_view = QtWidgets.QAction("Show File View", self)
        action_show_file_view.triggered.connect(self.file_view_dock.show)
        menu_view.addAction(action_show_file_view)

        action_show_network_view = QtWidgets.QAction("Show Network View", self)
        action_show_network_view.triggered.connect(self.network_view_dock.show)
        menu_view.addAction(action_show_network_view)


if __name__ == '__main__':

    app = QtWidgets.QApplication(sys.argv)
    main_window = MainWindow()
    main_window.setGeometry(50, 50, 1200, 700)
    main_window.setWindowTitle("Attention Editor")
    main_window.show()
    sys.exit(app.exec_())

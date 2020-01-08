#!/usr/bin/env python
# -*- coding: utf-8 -*-

# image_view.py
# Copyright (c) Tsubasa Hirakawa, 2020


import numpy as np
import cv2
from PyQt5 import QtWidgets, QtCore, QtGui

from atteditor.utils import min_max


class ImageView(QtWidgets.QWidget):

    def __init__(self):
        super(ImageView, self).__init__()

        self.orig_image = None
        self.HEIGHT = None
        self.WIDTH = None
        self.CHANNEL = None
        self.att_map = None
        self.ATT_SIZE = (14, 14)

        # image
        self.image = QtGui.QImage()
        self.pixel_map = QtGui.QPixmap()
        self.label = QtWidgets.QLabel(self)
        self.label.setAlignment(QtCore.Qt.AlignCenter)

        # radio button
        self.button_add = QtWidgets.QRadioButton("add")
        self.button_remove = QtWidgets.QRadioButton("remove")
        self.button_add.setChecked(True)

        # slider
        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider.setRange(1, 50)
        self.slider.setValue(10)
        self.slider.valueChanged.connect(self.update_pen_mask)
        self.pen_size_label = QtWidgets.QLabel("Size: %d" % self.slider.value() * 2)

        # paint
        self.pen_size = self.slider.value()
        self.pen_mask = np.empty(0, dtype=bool)
        self.update_pen_mask()

        # layout
        _button_layout = QtWidgets.QHBoxLayout()
        _button_layout.addWidget(self.button_add)
        _button_layout.addWidget(self.button_remove)
        _slider_layout = QtWidgets.QHBoxLayout()
        _slider_layout.addWidget(self.pen_size_label)
        _slider_layout.addWidget(self.slider)
        _layout = QtWidgets.QVBoxLayout()
        _layout.addWidget(self.label)
        _layout.addLayout(_button_layout)
        _layout.addLayout(_slider_layout)
        self.setLayout(_layout)

    def update_pen_mask(self):
        self.pen_size = self.slider.value()
        c = self.pen_size
        r = self.pen_size
        y, x = np.ogrid[-c:self.pen_size*2-c, -c:self.pen_size*2-c]
        mask = x*x + y*y <= r*r
        self.pen_mask = np.zeros((self.pen_size*2, self.pen_size*2), dtype=np.bool)
        self.pen_mask[mask] = True

        self.pen_size_label.setText("Size: %d" % (self.pen_size * 2))

    def reset_image(self, image):
        self.att_map = None
        self.orig_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.HEIGHT, self.WIDTH, self.CHANNEL = self.orig_image.shape
        self.image = QtGui.QImage(self.orig_image.data,
                                  self.WIDTH,
                                  self.HEIGHT,
                                  self.WIDTH * self.CHANNEL,
                                  QtGui.QImage.Format_RGB888)
        self.pixel_map = QtGui.QPixmap(QtGui.QPixmap.fromImage(self.image).
                                       scaledToWidth(self.WIDTH))
        self.label.setPixmap(self.pixel_map)

    def update_attention_image(self, input_att_map):
        self.att_map = cv2.resize(input_att_map, (self.WIDTH, self.HEIGHT))
        _att_tmp = min_max(self.att_map) * 255.
        _jet_map = cv2.applyColorMap(_att_tmp.astype(np.uint8), cv2.COLORMAP_JET)

        _image_tmp = cv2.addWeighted(self.orig_image, 0.5, _jet_map, 0.5, 0)
        _image_tmp = cv2.cvtColor(_image_tmp, cv2.COLOR_RGB2BGR)

        self.image = QtGui.QImage(_image_tmp.data,
                                  self.WIDTH,
                                  self.HEIGHT,
                                  self.WIDTH * self.CHANNEL,
                                  QtGui.QImage.Format_RGB888)
        self.pixel_map = QtGui.QPixmap(QtGui.QPixmap.fromImage(self.image).
                                       scaledToWidth(self.WIDTH))
        self.label.setPixmap(self.pixel_map)

    def get_attention_map(self):
        if self.att_map is None:
            return None
        else:
            return cv2.resize(self.att_map, self.ATT_SIZE)

    def edit_attention_map(self, x, y):

        _x_min = x - self.pen_size
        _x_max = x + self.pen_size
        _y_min = y - self.pen_size
        _y_max = y + self.pen_size

        mask = self.pen_mask.copy()
        if _y_min < 0:
            mask = mask[abs(_y_min):, :]
            _y_min = 0
        if _y_max >= self.HEIGHT:
            mask = mask[:-abs(self.HEIGHT - _y_max), :]
        if _x_min < 0:
            mask = mask[:, abs(_x_min):]
            _x_min = 0
        if _x_max >= self.WIDTH:
            mask = mask[:, :-abs(self.WIDTH - _x_max)]

        _sub_att = self.att_map[_y_min:_y_max, _x_min:_x_max]

        if self.button_add.isChecked():
            _sub_att[mask] = 1.0
        else:
            _sub_att[mask] = 0.0

    def mousePressEvent(self, QMouseEvent):
        if self.orig_image is not None and self.att_map is not None:
            _point_x = int((QMouseEvent.pos().x() - self.label.x()) - \
                           (self.label.frameRect().width() - self.pixel_map.width()) / 2)
            _point_y = int((QMouseEvent.pos().y() - self.label.y()) - \
                           (self.label.frameRect().height() - self.pixel_map.height()) / 2)

            if self.check_exceeded_point(_point_x, _point_y):
                self.edit_attention_map(_point_x, _point_y)
                self.update_attention_image(self.att_map)

        self.update()

    def mouseMoveEvent(self, QMouseEvent):
        if self.orig_image is not None and self.att_map is not None:
            _point_x = int((QMouseEvent.pos().x() - self.label.x()) - \
                           (self.label.frameRect().width() - self.pixel_map.width()) / 2)
            _point_y = int((QMouseEvent.pos().y() - self.label.y()) - \
                           (self.label.frameRect().height() - self.pixel_map.height()) / 2)

            if self.check_exceeded_point(_point_x, _point_y):
                self.edit_attention_map(_point_x, _point_y)
                self.update_attention_image(self.att_map)

        self.update()

    def check_exceeded_point(self, x, y):
        if 0 <= x <= self.WIDTH:
            in_x = True
        else:
            in_x = False
        if 0 <= y <= self.HEIGHT:
            in_y = True
        else:
            in_y = False
        return bool(in_x * in_y)


if __name__ == '__main__':
    import sys

    IMG_NAME = "/Users/hirakawa/Dataset/CUB-200-2010/images/" \
               "001.Black_footed_Albatross/" \
               "Black_footed_Albatross_0001_2950163169.jpg"
    img = cv2.imread(IMG_NAME)

    app = QtWidgets.QApplication(sys.argv)
    widget = ImageView()
    widget.load_image(img)
    widget.setGeometry(100, 100, 500, 500)
    widget.show()
    sys.exit(app.exec_())

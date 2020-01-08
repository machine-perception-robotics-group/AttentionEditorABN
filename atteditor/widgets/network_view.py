#!/usr/bin/env python
# -*- coding: utf-8 -*-

# network_view.py
# Copyright (c) Tsubasa Hirakawa, 2020


import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torchvision import transforms

from ..models.cub import *
from ..utils import softmax
from ..config import MODEL_NAME, CHECKPOINT_FILE

from PyQt5 import QtWidgets, QtGui, QtCore


class NetworkView(QtWidgets.QWidget):

    def __init__(self):
        super(NetworkView, self).__init__()

        # gpu and network model settings ##################
        # gpu
        self.use_cuda = torch.cuda.is_available()

        # set network model
        self.model_name = MODEL_NAME
        self.model = cub_resnet50(CHECKPOINT_FILE)
        if self.use_cuda:
            self.model = nn.DataParallel(self.model).cuda()
            cudnn.benchmark = True
        self.model.eval()

        # softmax and preprocess
        self.preprocess = transforms.Compose([transforms.ToTensor(),
                                              transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                                                   std=[0.5, 0.5, 0.5])])

        # class label (list)
        self.class_label = class_label
        self.n_class = len(self.class_label)
        ###################################################

        self.label_model_name = QtWidgets.QLabel("")
        self.button_infer = QtWidgets.QPushButton("infer")
        self.button_infer.setSizePolicy(QtWidgets.QSizePolicy.Fixed,
                                        QtWidgets.QSizePolicy.Fixed)

        # score view
        self.n_display = 20
        if self.n_display < self.n_class:
            self.n_display = self.n_class
        self.score_title = QtWidgets.QLabel("Top-%d results" % self.n_display)
        self.score_model = QtGui.QStandardItemModel(0, 3)
        self.score_model.setHeaderData(0, QtCore.Qt.Horizontal, 'index')
        self.score_model.setHeaderData(1, QtCore.Qt.Horizontal, 'name')
        self.score_model.setHeaderData(2, QtCore.Qt.Horizontal, 'score')
        self.score = QtWidgets.QTreeView()
        self.score.setModel(self.score_model)

        _layout = QtWidgets.QVBoxLayout()
        _layout.addWidget(self.label_model_name)
        _layout.addWidget(self.button_infer)
        _layout.addWidget(self.score_title)
        _layout.addWidget(self.score)

        self.setLayout(_layout)

        self.set_label()

    def set_label(self):
        self.label_model_name.setText("Model: " + self.model_name)

    def infer(self, image, input_att_map=None):
        _x = self.preprocess(image)
        _x = torch.unsqueeze(_x, 0)

        if input_att_map is not None:
            input_att_map = torch.from_numpy(input_att_map[np.newaxis, np.newaxis, :])

        if self.use_cuda:
            _x = _x.cuda()
            if input_att_map is not None:
                input_att_map = input_att_map.cuda()

        with torch.no_grad():
            _, out_per, att_map = self.model(_x, input_att_map)

        if self.use_cuda:
            out_per = out_per.data.cpu().numpy()
            att_map = att_map.data.cpu().numpy()
        else:
            out_per = out_per.data.numpy()
            att_map = att_map.data.numpy()

        # score update
        self.update_score(out_per)

        # return
        return att_map[0, 0, :, :]

    def update_score(self, input_score):
        _prob = softmax(input_score)
        _ordered_index = np.argsort(-input_score)

        self.clear_score()
        for i in range(self.n_display):
            _index = _ordered_index[0, i]
            self.score_model.setItem(i, 0, QtGui.QStandardItem("%04d" % _index))
            self.score_model.setItem(i, 1, QtGui.QStandardItem(self.class_label[_index]))
            self.score_model.setItem(i, 2, QtGui.QStandardItem("%f" % _prob[0, _index]))

    def clear_score(self):
        self.score_model.removeRows(0, self.score_model.rowCount())


if __name__ == '__main__':
    import sys

    app = QtWidgets.QApplication(sys.argv)
    widget = NetworkView()
    widget.setGeometry(100, 100, 300, 300)
    widget.show()
    sys.exit(app.exec_())

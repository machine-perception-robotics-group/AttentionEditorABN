#!/usr/bin/env python
# -*- coding: utf-8 -*-

# utils.py
# Copyright (c) Tsubasa Hirakawa, 2020


import numpy as np


def center_crop(image, size=224):
    _h, _w, _c = image.shape
    _h_min = int(_h / 2) - int(size / 2)
    _h_max = int(_h / 2) + int(size / 2)
    _w_min = int(_w / 2) - int(size / 2)
    _w_max = int(_w / 2) + int(size / 2)
    _dst = image[_h_min:_h_max, _w_min:_w_max, :]
    return _dst


def softmax(x):
    x_max = np.max(x)
    x = np.exp(x - x_max)
    u = np.sum(x)
    return x / u


def min_max(x, axis=None):
    min = 0  #x.min(axis=axis, keepdims=True)
    max = 1  #x.max(axis=axis, keepdims=True)
    result = (x - min) / (max - min)
    return result

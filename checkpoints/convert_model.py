#!/usr/bin/env python
# -*- coding: utf-8 -*-

# convert_model.py
# Copyright (c) Tsubasa Hirakawa, 2020


from argparse import ArgumentParser
from collections import OrderedDict
import torch


def parser():
    arg_parser = ArgumentParser(add_help=True)

    arg_parser.add_argument('load_model')
    arg_parser.add_argument('save_model')

    args = arg_parser.parse_args()
    return args


def main():
    args = parser()

    # open original model file
    params = torch.load(args.load_model, map_location='cpu')
    state_dict = params['state_dict']

    # convert model params (remove "module.")
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove "module."
        new_state_dict[name] = v

    # save model
    torch.save(new_state_dict, args.save_model)


if __name__ == '__main__':
    main()

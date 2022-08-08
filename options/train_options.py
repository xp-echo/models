#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import argparse

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from lib import NervusLogger

logger = NervusLogger.get_logger('options.train_options')

class TrainOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser(description='Train options')

        # Materials
        self.parser.add_argument('--csv_name',  type=str, default=None, help='csv filename(Default: None)')

        # Task
        self.parser.add_argument('--task',  type=str, default=None, help='Task: classification or regression (Default: None)')

        # Model
        self.parser.add_argument('--model', type=str, default=None, help='model: CNN name')

        # Training and Internal validation
        self.parser.add_argument('--criterion', type=str,   default=None,               help='criterion: CEL, MSE, RMSE, MAE (Default: None)')
        self.parser.add_argument('--optimizer', type=str,   default=None,               help='optimzer:SGD, Adadelta, Adam, RMSprop (Default: None)')
        self.parser.add_argument('--lr',        type=float, default=0.001, metavar='N', help='learning rate: (Default: 0.001)')
        self.parser.add_argument('--epochs',    type=int,   default=10,    metavar='N', help='number of epochs (Default: 10)')

        # Batch size
        self.parser.add_argument('--batch_size', type=int, default=None, metavar='N', help='batch size for training (Default: None)')

        # Preprocess for image
        self.parser.add_argument('--augmentation',           type=str, default=None,  help='Apply all augumentation except normalize_image, yes or no (Default: None)')
        self.parser.add_argument('--normalize_image',        type=str, default='yes', help='image nomalization, yes no no (Default: yes)')

        # Input channel
        self.parser.add_argument('--input_channel', type=int, default=None, help='channel of input image (Default: None)')

        # Weight saving strategy
        self.parser.add_argument('--save_weight', type=str, choices=['best', 'each'], default='best', help='Save weight: best, or each time loss decreases when multi-label output(Default: None)')

        # GPU
        self.parser.add_argument('--gpu_ids', type=str, default='-1', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU (Default: -1)')


    def _get_args(self):
        return vars(self.args)


    def parse(self):
        self.args = self.parser.parse_args()

        self.args.cnn = self.args.model

        # Align options for augmentation
        if self.args.augmentation is not None:
            self.args.preprocess = 'yes'
        else:
            self.args.preprocess = 'no'

        # Align gpu_ids
        str_ids = self.args.gpu_ids.split(',')
        self.args.gpu_ids = []

        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                self.args.gpu_ids.append(id)

        return self._get_args()


    def print_options(self):
        message = ''
        message += '-------------------- Options --------------------\n'

        for k, v in (self._get_args().items()):
            comment = ''
            default = self.parser.get_default(k)

            str_default = str(default) if str(default) != '' else 'None'
            str_v = str(v) if str(v) != '' else 'Not specified'

            if k == 'gpu_ids':
                if str_v == '[]':
                    str_v = 'CPU selected'
                else:
                    str_v = str_v + ' (Primary GPU:{})'.format(v[0])

            comment = ('\t[Default: %s]' % str_default) if k != 'gpu_ids' else '\t[Default: CPU]'
            message += '{:>25}: {:<30}{}\n'.format(str(k), str_v, comment)

        message += '------------------- End --------------------------'
        logger.info(message)

#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys

import torch
import torch.nn as nn

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from lib import NervusLogger

logger = NervusLogger.get_logger('config.criterion')

class RMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, yhat, y):
        return torch.sqrt(self.mse(yhat, y))


def set_criterion(criterion_name):
    if criterion_name == 'CEL':
        criterion = nn.CrossEntropyLoss()

    elif criterion_name == 'MSE':
        criterion = nn.MSELoss()

    elif criterion_name == 'RMSE':
        criterion = RMSELoss()

    elif criterion_name == 'MAE':
        criterion = nn.L1Loss()

    else:
        logger.error(f"No specified criterion: {criterion_name}.")

    return criterion

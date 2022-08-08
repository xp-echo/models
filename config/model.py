#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
from collections import OrderedDict

import torch
import torch.nn as nn
import torchvision.models as models

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from lib import *

logger = NervusLogger.get_logger('config.model')



DUMMY_LAYER = nn.Identity()


# For ResNet family
class ResNet_Multi(nn.Module):
    def __init__(self, base_model, label_num_classes):
        super().__init__()
        self.extractor = base_model
        self.label_num_classes = label_num_classes
        self.label_list = list(label_num_classes.keys())
        _prefix_layer = 'fc_'
        self.fc_names = [(_prefix_layer + label_name) for label_name in self.label_list]

        # Construct fc layres
        self.input_size_fc = self.extractor.fc.in_features
        self.fc_multi = nn.ModuleDict({
                            (_prefix_layer + label_name): nn.Linear(self.input_size_fc, num_outputs)
                            for label_name, num_outputs in self.label_num_classes.items()
                        })

        # Replace the original fc layer
        self.extractor.fc = DUMMY_LAYER

    def forward(self, x):
        x = self.extractor(x)
        # Fork forwarding
        output_multi = {fc_name: self.fc_multi[fc_name](x) for fc_name in self.fc_names}
        return output_multi


# For DenseNet family
class DenseNet_Multi(nn.Module):
    def __init__(self, base_model, label_num_classes):
        super().__init__()
        self.extractor = base_model
        self.label_num_classes = label_num_classes
        self.label_list = list(label_num_classes.keys())
        _prefix_layer = 'fc_'
        self.fc_names = [(_prefix_layer + label_name) for label_name in self.label_list]

        # Construct fc layres
        self.input_size_fc = self.extractor.classifier.in_features
        self.fc_multi = nn.ModuleDict({
                            (_prefix_layer + label_name): nn.Linear(self.input_size_fc, num_outputs)
                            for label_name, num_outputs in self.label_num_classes.items()
                        })

        # Replace the original fc layer
        self.extractor.classifier = DUMMY_LAYER

    def forward(self, x):
        x = self.extractor(x)
        # Fork forwarding
        output_multi = {fc_name: self.fc_multi[fc_name](x) for fc_name in self.fc_names}
        return output_multi


# For EfficientNet family
class EfficientNet_Multi(nn.Module):
    def __init__(self, base_model, label_num_classes):
        super().__init__()
        self.extractor = base_model
        self.label_num_classes = label_num_classes
        self.label_list = list(label_num_classes.keys())
        _prefix_layer = 'block_'
        self.block_names = [(_prefix_layer + label_name) for label_name in self.label_list]

        # Construct fc layres
        _probability_dropout = self.extractor.classifier[0].p
        _input_size_fc = self.extractor.classifier[1].in_features
        self.fc_multi = nn.ModuleDict({
                            (_prefix_layer + label_name): nn.Sequential(
                                                                OrderedDict([
                                                                    ('0_' + label_name, nn.Dropout(p=_probability_dropout, inplace=False)),
                                                                    ('1_' + label_name, nn.Linear(_input_size_fc, num_outputs))
                                                                ]))
                            for label_name, num_outputs in self.label_num_classes.items()
                        })

        # Replace the original classifier
        self.extractor.classifier = DUMMY_LAYER

    def forward(self, x):
        x = self.extractor(x)
        # Fork forwarding
        output_multi = {block_name: self.fc_multi[block_name](x) for block_name in self.block_names}
        return output_multi


def set_model(cnn_name):
    if cnn_name == 'B0':
        cnn = models.efficientnet_b0

    elif cnn_name == 'B2':
        cnn = models.efficientnet_b2

    elif cnn_name == 'B4':
        cnn = models.efficientnet_b4

    elif cnn_name == 'B6':
        cnn = models.efficientnet_b6

    elif cnn_name == 'ResNet18':
        cnn = models.resnet18

    elif cnn_name == 'ResNet':
        cnn = models.resnet50

    elif cnn_name == 'DenseNet':
        cnn = models.densenet161

    else:
        logger.error(f"No such a specified CNN: {cnn_name}.")

    return cnn


# Change input channle of CNN to 1ch
def align_1ch_channel(cnn_name, cnn):
    if cnn_name.startswith('ResNet'):
        cnn.conv1.in_channels = 1
        cnn.conv1.weight = nn.Parameter(cnn.conv1.weight.sum(dim=1).unsqueeze(1))

    elif cnn_name.startswith('B'):
        cnn.features[0][0].in_channels = 1
        cnn.features[0][0].weight = nn.Parameter(cnn.features[0][0].weight.sum(dim=1).unsqueeze(1))

    elif cnn_name.startswith('DenseNet'):
        cnn.features.conv0.in_channels = 1
        cnn.features.conv0.weight = nn.Parameter(cnn.features.conv0.weight.sum(dim=1).unsqueeze(1))

    else:
        logger.error(f"No such a specified CNN: {cnn_name}.")

    return cnn


def conv_net(cnn_name, label_num_classes, input_channel):
    cnn = set_model(cnn_name)

    # Once make pseudo-model for 3ch and single.
    label_list = list(label_num_classes.keys())
    num_outputs_first_label = label_num_classes[label_list[0]]

    cnn = cnn(num_classes=num_outputs_first_label)

    # Align 1ch or 3ch
    if input_channel == 1:
        cnn = align_1ch_channel(cnn_name, cnn)
    else:
        cnn = cnn

    # Make multi
    if cnn_name.startswith('ResNet'):
        cnn = ResNet_Multi(cnn, label_num_classes)

    elif cnn_name.startswith('B'):
        cnn = EfficientNet_Multi(cnn, label_num_classes)

    elif cnn_name.startswith('DenseNet'):
        cnn = DenseNet_Multi(cnn, label_num_classes)

    else:
        logger.error(f"Cannot make multi: {cnn_name}.")

    return cnn


def create_cnn(cnn, label_num_classes, input_channel, gpu_ids=[]):
    model = conv_net(cnn, label_num_classes, input_channel)
    device = set_device(gpu_ids)
    model.to(device)
    if gpu_ids:
        model = torch.nn.DataParallel(model, gpu_ids)
    else:
        pass
    return model


# Extract outputs of label_name
def get_layer_output(outputs_multi, label_name):
    output_layer_names = outputs_multi.keys()
    layer_name = [output_layer_name for output_layer_name in output_layer_names if output_layer_name.endswith(label_name)][0]
    output_layer = outputs_multi[layer_name]
    return output_layer

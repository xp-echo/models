#!/usr/bin/env python
# -*- coding: utf-8 -*-

from abc import ABC, abstractmethod
import os
import sys
import numpy as np
import pandas as pd

import torch
import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
from PIL import Image

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from lib.util import *


class NervusDataSet(Dataset, ABC):
    def __init__(self, args, split_provider, split_list, multi_label=False):
        super().__init__()

        self.args = args
        self.split_provider = split_provider
        self.split_list = split_list

        self.df_source = self.split_provider.df_source
        self.id_column = self.split_provider.id_column
        self.institution_column = self.split_provider.institution_column
        self.examid_column = self.split_provider.examid_column

        self.raw_label_name = self.split_provider.raw_label_list[0]
        self.internal_label_name = self.split_provider.internal_label_list[0]

        self.filepath_column = self.split_provider.filepath_column
        self.split_column = self.split_provider.split_column
        self.df_split = get_column_value(self.df_source, self.split_column, self.split_list)

        self.transform = self._make_transforms()
        self.augmentation = self._make_augmentations()

        # index of each column
        self.index_dict = {column_name: self.df_split.columns.get_loc(column_name) for column_name in self.df_split.columns}


    def _make_transforms(self):
        _transforms = []
        _transforms.append(transforms.ToTensor())

        if self.args['normalize_image'] == 'yes':
            if self.args['input_channel'] == 1:
                _transforms.append(transforms.Normalize(mean=(0.5, ), std=(0.5, )))

            elif self.args['input_channel'] == 3:
                _transforms.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))

            else:
                logger.error(f"Invalid input channel: {self.args['input_channel']}.")

        _transforms = transforms.Compose(_transforms)
        return _transforms

    def _make_augmentations(self):
        _augmentation = []

        if self.args['preprocess'] == 'yes':
            if self.args['augmentation'] == 'randaug':
                _augmentation.append(transforms.RandAugment())

            elif self.args['augmentation'] == 'trivialaugwide':
                _augmentation.append(transforms.TrivialAugmentWide())

            elif self.args['augmentation'] == 'augmix':
                _augmentation.append(transforms.AugMix())

            else:
                pass

        _augmentation = transforms.Compose(_augmentation)
        return _augmentation


    def __len__(self):
        return len(self.df_split)


    # Load imgae when CNN or MLP+CNN
    def _load_image_if_cnn(self, idx):
        image = ""

        if self.args["cnn"] is None:
            return image

        image_path = os.path.join('./materials/images', self.df_split.iat[idx, self.index_dict[self.filepath_column]])

        assert (self.args['input_channel'] == 1) or (self.args['input_channel'] == 3), f"Invalid input channel: {self.args['input_channel']}."
        if self.args['input_channel'] == 1:
            image = Image.open(image_path).convert('L')
        else:
            image = Image.open(image_path).convert('RGB')

        image = self.augmentation(image)
        image = self.transform(image)

        return image


    @abstractmethod
    def __getitem__(self, idx):
        pass


    @classmethod
    @abstractmethod
    def create_dataloader(cls, args, csv_dict, images_dir, split_list=None, batch_size=None):
        pass

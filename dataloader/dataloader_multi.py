#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys

from torch.utils.data.dataloader import DataLoader

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from lib.util import *
from dataloader.nervus_dataloader import NervusDataSet

class MultiLabelDataSet(NervusDataSet):
    def __init__(self, args, split_provider, split_list):
        multi_label = True
        super().__init__(args, split_provider, split_list, multi_label)

    def __getitem__(self, idx):
        id = self.df_split.iat[idx, self.index_dict[self.id_column]]
        institution = self.df_split.iat[idx, self.index_dict[self.institution_column]]
        examid = self.df_split.iat[idx, self.index_dict[self.examid_column]]
        raw_label_dict = {row_label_name: self.df_split.iat[idx, self.index_dict[row_label_name]] for row_label_name in self.split_provider.raw_label_list}
        internal_label_dict = {internal_label_name: self.df_split.iat[idx, self.index_dict[internal_label_name]] for internal_label_name in self.split_provider.internal_label_list}
        split = self.df_split.iat[idx, self.index_dict[self.split_column]]

        image = self._load_image_if_cnn(idx)

        return id, institution, examid, raw_label_dict, internal_label_dict, image, split


    @classmethod
    def create_dataloader(cls, args, split_provider, split_list=None, batch_size=None):
        assert (split_list is not None), 'Specify split to make dataloader.'

        split_data = cls(args, split_provider, split_list)
        split_loader = DataLoader(
                                dataset = split_data,
                                batch_size = batch_size,
                                shuffle = True,
                                num_workers = 0)
        return split_loader

#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
import dataclasses


@dataclasses.dataclass
class NervusEnv:
    dataroot: str = './materials'
    splits_dir: str = os.path.join(dataroot, 'splits')
    images_dir: str = os.path.join(dataroot, 'images')
    results_dir: str = './results'
    sets_dir: str = os.path.join(results_dir, 'sets')

    weight_dir: str = 'weights'
    learning_curve_dir: str = 'learning_curves'
    likelihood_dir: str = 'likelihoods'

    csv_name_parameters: str = 'parameter.csv'
    weight_name: str = 'weight'
    csv_name_learning_curve: str = 'learning_curve'
    csv_name_likelihood: str = 'likelihood'

    summary_dir: str = os.path.join(results_dir, 'summary')
    csv_summary: str = 'summary.csv'


class SplitProvider:
    def __init__(self, split_path, task):
        super().__init__()

        self.split_path = split_path
        self.task = task

        self.prefix_id = 'id'
        self.prefix_raw_label = 'label'
        self.prefix_internal_label = 'internal_label'
        self.prefix_period = 'periods'

        self.institution_column = 'Institution'
        self.examid_column = 'ExamID'
        self.filepath_column = 'filepath'
        self.split_column = 'split'

        _df_source = pd.read_csv(self.split_path)
        _df_source_excluded = _df_source[_df_source[self.split_column] != 'exclude'].copy()

        # Labelling
        _df_source_labeled, _class_name_in_raw_label = self._make_labelling(_df_source_excluded, self.task)

        # Cast
        self.df_source = self._cast_csv(_df_source_labeled, self.task)
        self.raw_label_list = list(self.df_source.columns[self.df_source.columns.str.startswith(self.prefix_raw_label)])
        self.internal_label_list = list(self.df_source.columns[self.df_source.columns.str.startswith(self.prefix_internal_label)])
        self.class_name_in_raw_label = _class_name_in_raw_label
        self.num_classes_in_internal_label = self._define_num_classes_in_internal_label(self.df_source, self.task)  
        self.id_column = list(self.df_source.columns[self.df_source.columns.str.startswith(self.prefix_id)])[0]


    # Labeling
    def _make_labelling(self, df_source_excluded, task):
        _df_tmp = df_source_excluded.copy()
        _raw_label_list = list(_df_tmp.columns[_df_tmp.columns.str.startswith(self.prefix_raw_label)])
        _class_name_in_raw_label = {}
        for raw_label_name in _raw_label_list:
            class_list = _df_tmp[raw_label_name].value_counts().index.tolist()
            _class_name_in_raw_label[raw_label_name] = {}
            if task == 'classification':
                for i in range(len(class_list)):
                    _class_name_in_raw_label[raw_label_name][class_list[i]] = i
            else:
                _class_name_in_raw_label[raw_label_name] = {}

        for raw_label_name, class_name_in_raw_label in _class_name_in_raw_label.items():
            _internal_label = self.prefix_internal_label + raw_label_name.replace(self.prefix_raw_label, '')
            if task == 'classification':
                for class_name, ground_truth in class_name_in_raw_label.items():
                    _df_tmp.loc[_df_tmp[raw_label_name]==class_name, _internal_label] = ground_truth
            else:
                # When regression
                _df_tmp[_internal_label] = _df_tmp[raw_label_name]

        _df_source_labeled = _df_tmp.copy()
        return _df_source_labeled, _class_name_in_raw_label


    def _define_num_classes_in_internal_label(self, df_source, task):
        _num_classes_in_internal_label = {}
        _internal_label_list = list(df_source.columns[df_source.columns.str.startswith(self.prefix_internal_label)])
        for internal_label_name in _internal_label_list:
            if task == 'classification':
                _num_classes_in_internal_label[internal_label_name] = df_source[internal_label_name].nunique()
            else:
                # When regression
                _num_classes_in_internal_label[internal_label_name] = 1

        return _num_classes_in_internal_label


    # Cast
    def _cast_csv(self, df_source_labeled, task):
        _df_tmp = df_source_labeled.copy()
        _internal_label_list = list(_df_tmp.columns[_df_tmp.columns.str.startswith(self.prefix_internal_label)])

        if task == 'classification':
            _cast_internal_label_dict = {internal_label: int for internal_label in _internal_label_list}
        else:
            # When regression
            _cast_internal_label_dict = {internal_label: float for internal_label in _internal_label_list}

        _df_tmp = _df_tmp.astype(_cast_internal_label_dict)

        _df_casted = _df_tmp.copy()
        return _df_casted

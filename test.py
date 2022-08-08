#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import glob
from typing import Tuple

import pandas as pd
import torch
from torch.utils.data.dataset import Dataset

import dataloader
from config.model import *
from lib import *
from options import TestOptions

logger = NervusLogger.get_logger('test')


nervusenv = NervusEnv()
args = TestOptions().parse()
datetime_dir = get_target(nervusenv.sets_dir, args['test_datetime'])
parameters_path = os.path.join(datetime_dir, nervusenv.csv_name_parameters)
train_parameters = read_train_parameters(parameters_path)
task = train_parameters['task']
cnn = train_parameters['cnn']
input_channel = train_parameters['input_channel']
gpu_ids = str2int(train_parameters['gpu_ids'])
device = set_device(gpu_ids)

sp = SplitProvider(os.path.join(nervusenv.splits_dir, train_parameters['csv_name']), task)
label_list = sp.internal_label_list

# Align option for test only
test_batch_size = args['test_batch_size']
train_parameters['preprocess'] = 'no'
train_parameters['normalize_image'] = args['normalize_image']


dataset_handler = dataloader.MultiLabelDataSet
def _execute_test(*args):
    return _execute_test_multi_label(*args)

train_loader = dataset_handler.create_dataloader(train_parameters, sp, split_list=['train'], batch_size=test_batch_size)
val_loader = dataset_handler.create_dataloader(train_parameters, sp, split_list=['val'], batch_size=test_batch_size)
test_loader = dataset_handler.create_dataloader(train_parameters, sp, split_list=['test'], batch_size=test_batch_size)

# Configure of model
model = create_cnn(cnn, sp.num_classes_in_internal_label, input_channel, gpu_ids=gpu_ids)


# Make column name of
column_pred_names_in_label_dict = {}
for raw_label_name, class_dict in sp.class_name_in_raw_label.items():
    pred_names = []
    if task == 'classification':
        for class_name in class_dict.keys():
            pred_names.append('pred_' + raw_label_name + '_' + str(class_name))
    else:
    # When regression
        pred_names.append('pred_' + raw_label_name)
    column_pred_names_in_label_dict[raw_label_name] = pred_names


def execute(test_weight_path):
    weight = torch.load(test_weight_path)
    model.load_state_dict(weight)
    model.eval()
    with torch.no_grad():
        train_acc = 0.0
        val_acc = 0.0
        test_acc = 0.0
        df_result = pd.DataFrame([])

        for _split in ['train', 'val', 'test']:
            if _split == 'train':
                _dataloader = train_loader
            elif _split == 'val':
                _dataloader = val_loader
            elif _split == 'test':
                _dataloader = test_loader
            else:
                logger.error('Split in dataloader error.')

            _train_acc, _val_acc, _test_acc, _df_result = _execute_test(_split, _dataloader, model)

            train_acc += _train_acc
            val_acc += _val_acc
            test_acc += _test_acc
            df_result = pd.concat([df_result, _df_result], ignore_index=True)

    return train_acc, val_acc, test_acc, df_result



def _execute_test_multi_label(split:str, dataloader:Dataset, model) -> Tuple[float, float, pd.DataFrame]:
    train_acc = 0.0
    val_acc = 0.0
    test_acc = 0.0
    df_result = pd.DataFrame([])

    for i, (ids, institutions, examids, raw_labels_dict, labels_dict, images, splits) in enumerate(dataloader):
        images = images.to(device)
        outputs = model(images)

        likelihood_multi = {}
        preds_multi = {}
        labels_multi = {label_name: labels.to(device) for label_name, labels in labels_dict.items()}
        for label_name, labels in labels_multi.items():
            likelihood_multi[label_name] = get_layer_output(outputs, label_name)   # No softmax
            if task == 'classification':
                preds_multi[label_name] = torch.max(likelihood_multi[label_name], 1)[1]
                split_acc_label_name = torch.sum(preds_multi[label_name] == labels.data).item()
                if split == 'train':
                    train_acc += split_acc_label_name
                elif split == 'val':
                    val_acc += split_acc_label_name
                elif split == 'test':
                    test_acc += split_acc_label_name
            else:
                pass

        labels_multi = {label_name: label.to('cpu').detach().numpy().copy() for label_name, label in labels_multi.items()}
        likelihood_multi = {label_name: likelihood.to('cpu').detach().numpy().copy() for label_name, likelihood in likelihood_multi.items()}

        df_id = pd.DataFrame({sp.id_column: ids})
        df_instituion = pd.DataFrame({sp.institution_column: institutions})
        df_examid = pd.DataFrame({sp.examid_column: examids})
        df_split = pd.DataFrame({sp.split_column: splits})
        df_likelihood_tmp = pd.DataFrame([])
        for label_name in likelihood_multi.keys():
            raw_label_name = label_name.split('_', 1)[-1]
            df_raw_label = pd.DataFrame(raw_labels_dict[raw_label_name], columns=[raw_label_name])
            df_likelihood = pd.DataFrame(likelihood_multi[label_name], columns=column_pred_names_in_label_dict[raw_label_name])
            df_likelihood_tmp = pd.concat([df_likelihood_tmp, df_raw_label, df_likelihood], axis=1)

        df_tmp = pd.concat([df_id, df_instituion, df_examid, df_likelihood_tmp, df_split], axis=1)
        df_result = pd.concat([df_result, df_tmp], ignore_index=True)

    return train_acc, val_acc, test_acc, df_result



if __name__=="__main__":
    test_weight_dir = os.path.join(datetime_dir, nervusenv.weight_dir)
    test_weight_path_list = sorted(glob.glob(test_weight_dir + '/*.pt'))

    for test_weight_path in test_weight_path_list:
        test_weight_name = os.path.basename(test_weight_path)
        # Inference
        logger.info(f"\nInference started ...")
        logger.info(f"weight: {test_weight_name}")
        train_total = len(train_loader.dataset)
        val_total = len(val_loader.dataset)
        test_total = len(test_loader.dataset)
        logger.info(f"train_data = {train_total}")
        logger.info(f"  val_data = {val_total}")
        logger.info(f" test_data = {test_total}")

        train_acc, val_acc, test_acc, df_result = execute(test_weight_path)

        if task == 'classification':
            train_acc = (train_acc / (train_total * len(label_list))) * 100
            val_acc = (val_acc / (val_total * len(label_list))) * 100
            test_acc = (test_acc / (test_total * len(label_list))) * 100
            logger.info(f"train: Inference_accuracy: {train_acc:.4f} %")
            logger.info(f"  val: Inference_accuracy: {val_acc:.4f} %")
            logger.info(f" test: Inference_accuracy: {test_acc:.4f} %")
        else:
            # When regresson
            pass
        logger.info('Inference finished!')

        # Save likelohood
        likelihood_dir = os.path.join(datetime_dir, nervusenv.likelihood_dir)
        os.makedirs(likelihood_dir, exist_ok=True)
        likelihood_name = nervusenv.csv_name_likelihood + '_' + test_weight_name.replace('.pt', '.csv')
        likelihood_path = os.path.join(likelihood_dir, likelihood_name)
        df_result.to_csv(likelihood_path, index=False)

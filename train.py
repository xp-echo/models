#!/usr/bin/env python
# -*- coding: utf-8 -*-

import datetime
import os
from typing import Tuple, Dict
import copy

import pandas as pd
import torch
import copy
from torch.utils.data.dataset import Dataset

import dataloader
from config import *
from lib import *
from options import TrainOptions

logger = NervusLogger.get_logger('train')


nervusenv = NervusEnv()
train_option_parser = TrainOptions()
args = train_option_parser.parse()
train_option_parser.print_options()

task = args['task']
cnn = args['cnn']
criterion = args['criterion']
optimizer = args['optimizer']
lr = args['lr']
num_epochs = args['epochs']
batch_size = args['batch_size']
input_channel = args['input_channel']
gpu_ids = args['gpu_ids']
save_weight = args['save_weight']
device = set_device(gpu_ids)


sp = SplitProvider(os.path.join(nervusenv.splits_dir, args['csv_name']), task)
label_list = sp.internal_label_list

dataset_handler = dataloader.MultiLabelDataSet
def _execute_task(*args):
    return _execute_multi_label(*args)

train_loader = dataset_handler.create_dataloader(args, sp, split_list=['train'], batch_size=batch_size)
val_loader = dataset_handler.create_dataloader(args, sp, split_list=['val'], batch_size=batch_size)

# Configure of training
model = create_cnn(cnn, sp.num_classes_in_internal_label, input_channel, gpu_ids=gpu_ids)
criterion = set_criterion(criterion)
optimizer = set_optimizer(optimizer, model, lr)


loss_acc_dict_classification = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
loss_acc_dict_regression = {'train_loss': [], 'val_loss': []}

# For overall loss and acc
loss_acc_dict = {}
if task == 'classification':
    loss_acc_dict = copy.deepcopy(loss_acc_dict_classification)
else:
    loss_acc_dict = copy.deepcopy(loss_acc_dict_regression)


# For label-wise loss and acc when multi label
if task == 'classification':
    loss_acc_label_wise_dict = {label_name: copy.deepcopy(loss_acc_dict_classification) for label_name in label_list}
else:
    loss_acc_label_wise_dict = {label_name: copy.deepcopy(loss_acc_dict_regression) for label_name in label_list}


best_weight = None
val_best_loss = None
val_best_epoch = None

val_best_loss_label_wise_dict = {label_name: None for label_name in label_list}
val_best_epoch_label_wise_dict = {label_name: None for label_name in label_list}


def execute(save_date_dir, save_weight, best_weight, val_best_loss, val_best_epoch, loss_acc_dict, val_best_loss_label_wise_dict, val_best_epoch_label_wise_dict, loss_acc_label_wise_dict, label_list):
    for _epoch in range(num_epochs):
        for _phase in ['train', 'val']:
            if _phase == 'train':
                model.train()
                _dataloader = train_loader
            else:
                model.eval()
                _dataloader = val_loader

            # Fot overall
            _running_loss = 0.0
            _running_acc = 0.0

            # For each label
            _running_loss_label_wise = None
            _running_acc_label_wise = None

            # execute task: execute_multi_label
            _running_loss, _running_acc, _running_loss_label_wise, _running_acc_label_wise = _execute_task(_phase, _dataloader)

            # Always update
            _update_flag = None
            loss_acc_dict, val_best_loss, val_best_epoch, _update_flag = _update_loss_acc_dict(task, num_epochs, loss_acc_dict, _epoch, _phase, _running_loss, _running_acc, len(_dataloader.dataset), len(label_list), val_best_loss, val_best_epoch)

            # Update only when multi-label
            _update_flag_label_wise = None
            loss_acc_label_wise_dict, val_best_loss_label_wise_dict, val_best_epoch_label_wise_dict, _update_flag_label_wise = _update_loss_acc_label_wise_dict(task, num_epochs, loss_acc_label_wise_dict, _epoch, _phase, _running_loss_label_wise, _running_acc_label_wise, len(_dataloader.dataset), 1, val_best_loss_label_wise_dict, val_best_epoch_label_wise_dict)


            # Keep weight each time epoch loss decreases.
            if (_phase == 'val' and _update_flag):
                best_weight = copy.deepcopy(model.state_dict())

                # Save weight each time epoch loss decreases only when multi-label.
                if (len(label_list) > 1) and (save_weight == 'each'):
                    if (0 < _epoch) and ((_epoch + 1) < num_epochs):
                        weight_name = nervusenv.weight_name + '_epoch-' + str(_epoch + 1).zfill(3) + '.pt'
                        weight_path = os.path.join(save_date_dir, nervusenv.weight_dir, weight_name)
                        torch.save(best_weight, weight_path)
                    else:
                        pass

    return best_weight, val_best_loss, val_best_epoch, loss_acc_dict, val_best_loss_label_wise_dict, val_best_epoch_label_wise_dict, loss_acc_label_wise_dict


def _execute_multi_label(phase:str, dataloader:Dataset) -> Tuple[float, float, Dict[str, float], Dict[str, float]]:
    # For overall
    running_loss = 0.0
    running_acc = 0.0

    # For each label
    running_loss_label_wise = {label_name : running_loss for label_name in label_list}
    running_acc_label_wise = {label_name : running_acc for label_name in label_list}

    for i, (ids, institutions, examids, raw_labels_dict, labels_dict, images, splits) in enumerate(dataloader):
        optimizer.zero_grad()

        with torch.set_grad_enabled(phase == 'train'):
            images = images.to(device)
            outputs = model(images)

            labels_multi = {label_name: labels.to(device) for label_name, labels in labels_dict.items()}

            # Initialize every iteration
            preds_multi = {}
            loss_multi = {}

            for label_name, labels in labels_multi.items():
                layer_outputs = get_layer_output(outputs, label_name)

                if task == 'classification':
                    preds_multi[label_name] = torch.max(layer_outputs, 1)[1]
                    loss_multi[label_name] = criterion(layer_outputs, labels)
                else:
                    loss_multi[label_name] = criterion(layer_outputs.squeeze(), labels.float())

            # Zero Reset
            loss = torch.tensor([0.0]).to(device)

            # Overall of loss for each label_i
            for loss_i in loss_multi.values():
                loss = torch.add(loss, loss_i)

            # Backward and Update weight
            if phase == 'train':
                loss.backward()
                optimizer.step()


        # Overall loss and acc
        if task == 'classification':
            running_loss += loss.item() * labels.size(0)

            for label_name, labels in labels_multi.items():
                running_acc_i = (torch.sum(preds_multi[label_name] == labels.data).item())
                running_acc += running_acc_i
        else:
            running_loss += loss.item() * labels.size(0)


        # Label-wise loss and acc
        for label_name, labels in labels_multi.items():
            if task == 'classification':
                running_loss_label_wise[label_name] += loss_multi[label_name].item() * labels.size(0)
                running_acc_label_wise[label_name] += (torch.sum(preds_multi[label_name] == labels.data).item())
            else:
                running_loss_label_wise[label_name] += loss_multi[label_name].item() * labels.size(0)

    return running_loss, running_acc, running_loss_label_wise, running_acc_label_wise


def _update_loss_acc_dict(task, num_epochs, loss_acc_dict, epoch, phase, running_loss, running_acc, len_dataloader, len_label_list, val_best_loss, val_best_epoch):
    update_comment = None
    update_flag = None

    if task == 'classification':
        epoch_loss = running_loss / (len_dataloader * len_label_list)
        epoch_acc = running_acc / (len_dataloader * len_label_list)
        if phase == 'train':
            loss_acc_dict['train_loss'].append(epoch_loss)
            loss_acc_dict['train_acc'].append(epoch_acc)
        else:
            loss_acc_dict['val_loss'].append(epoch_loss)
            loss_acc_dict['val_acc'].append(epoch_acc)
    else:
        # When regression
        epoch_loss = running_loss / (len_dataloader * len_label_list)
        if phase == 'train':
            loss_acc_dict['train_loss'].append(epoch_loss)
        else:
            loss_acc_dict['val_loss'].append(epoch_loss)

    # Check if val_best_loss
    if (phase == 'val') and ((val_best_loss is None) or (epoch_loss < val_best_loss)):
        val_best_loss = epoch_loss
        val_best_epoch = epoch + 1
        update_comment = ' Updated val_best_loss!'
        update_flag = True
    else:
        update_comment = ''
        update_flag = False

    # Print loss and acc at last epoch
    if phase == 'val':
        if task == 'classification':
            logger.info(f"epoch [{epoch+1:>3}/{num_epochs:<3}], train_loss: {loss_acc_dict['train_loss'][-1]:.4f}, val_loss: {loss_acc_dict['val_loss'][-1]:.4f}, val_acc: {loss_acc_dict['val_acc'][-1]:.4f}" + update_comment)
        else:
            # When regression
            logger.info(f"epoch [{epoch+1:>3}/{num_epochs:<3}], train_loss: {loss_acc_dict['train_loss'][-1]:.4f}, val_loss: {loss_acc_dict['val_loss'][-1]:.4f}" + update_comment)

    return loss_acc_dict, val_best_loss, val_best_epoch, update_flag


def _update_loss_acc_label_wise_dict(task, num_epochs, loss_acc_label_wise_dict, epoch, phase, running_loss_label_wise, running_acc_label_wise, len_dataloader, len_label_list, val_best_loss_label_wise_dict, val_best_epoch_label_wise_dict):
    update_flag_label_wise = {label_name: None for label_name in val_best_loss_label_wise_dict.keys()}

    # Update label-wise
    for label_name in loss_acc_label_wise_dict.keys():
        if task == 'classification':
            epoch_loss_each_label = running_loss_label_wise[label_name] / len_dataloader
            epoch_acc_each_label = running_acc_label_wise[label_name] / len_dataloader

            if phase == 'train':
                loss_acc_label_wise_dict[label_name]['train_loss'].append(epoch_loss_each_label)
                loss_acc_label_wise_dict[label_name]['train_acc'].append(epoch_acc_each_label)
            else:
                loss_acc_label_wise_dict[label_name]['val_loss'].append(epoch_loss_each_label)
                loss_acc_label_wise_dict[label_name]['val_acc'].append(epoch_acc_each_label)
        else:
            # When regression
            epoch_loss_each_label = running_loss_label_wise[label_name] / len_dataloader
            if phase == 'train':
                loss_acc_label_wise_dict[label_name]['train_loss'].append(epoch_loss_each_label)
            else:
                loss_acc_label_wise_dict[label_name]['val_loss'].append(epoch_loss_each_label)

        # Check if val_best_loss label-wise.
        if (phase == 'val') and ((val_best_loss_label_wise_dict[label_name] is None) or (epoch_loss_each_label < val_best_loss_label_wise_dict[label_name])):
                    val_best_loss_label_wise_dict[label_name] = epoch_loss_each_label
                    val_best_epoch_label_wise_dict[label_name] = epoch + 1
                    update_flag_label_wise[label_name] = True
        else:
            update_flag_label_wise[label_name] = False
    return loss_acc_label_wise_dict, val_best_loss_label_wise_dict, val_best_epoch_label_wise_dict, update_flag_label_wise




def save_result(save_date_dir, best_weight, val_best_loss, val_best_epoch, loss_acc_dict, val_best_loss_label_wise_dict, val_best_epoch_label_wise_dict, loss_acc_label_wise_dict):
    # Parameters
    df_opt = pd.DataFrame(list(args.items()), columns=['option', 'value'])
    parameters_path = os.path.join(save_date_dir, nervusenv.csv_name_parameters)
    df_opt.to_csv(parameters_path, index=False)

    # Weight
    weight_dir = os.path.join(save_date_dir, nervusenv.weight_dir)
    already_saved_weight_path = os.path.join(weight_dir, nervusenv.weight_name + '_epoch-' + str(val_best_epoch).zfill(3) + '.pt')
    # Is the best weight already been save?
    if os.path.exists(already_saved_weight_path):
        os.rename(already_saved_weight_path, already_saved_weight_path.replace('.pt', '-best.pt'))
    else:
        best_weight_name = nervusenv.weight_name + '_epoch-' + str(val_best_epoch).zfill(3) + '-best.pt'
        best_weight_path = os.path.join(weight_dir, best_weight_name)
        torch.save(best_weight, best_weight_path)

    # Learning curve
    learning_curve_dir = os.path.join(save_date_dir, nervusenv.learning_curve_dir)
    os.makedirs(learning_curve_dir, exist_ok=True)
    postfix_val_best = lambda label_name, val_best_epoch, val_best_loss: '_' + label_name + '_val-best-epoch-' + str(val_best_epoch).zfill(3) + '_val-best-loss-' + f"{val_best_loss:.4f}"

    # Overall learning curve
    learning_curve_overall_name = nervusenv.csv_name_learning_curve + postfix_val_best('overall', val_best_epoch, val_best_loss) + '.csv'
    learning_curve_overall_path = os.path.join(learning_curve_dir, learning_curve_overall_name)
    df_learning_curve = pd.DataFrame(loss_acc_dict)
    df_learning_curve.to_csv(learning_curve_overall_path, index=False)

    # Label-wise learning curve only when multi-label output
    if (loss_acc_label_wise_dict is not None):
        for label_name, loss_acc_dic_each_label in loss_acc_label_wise_dict.items():
            learning_curve_each_label_name = nervusenv.csv_name_learning_curve + postfix_val_best(label_name.replace(sp.prefix_internal_label, sp.prefix_raw_label), val_best_epoch_label_wise_dict[label_name], val_best_loss_label_wise_dict[label_name]) + '.csv'
            learning_curve_each_label_path = os.path.join(learning_curve_dir, learning_curve_each_label_name)
            df_learning_curve_each_label = pd.DataFrame(loss_acc_dic_each_label)
            df_learning_curve_each_label.to_csv(learning_curve_each_label_path, index=False)



if __name__=="__main__":
    date_now = datetime.datetime.now()
    date_name = date_now.strftime('%Y-%m-%d-%H-%M-%S')
    save_date_dir = os.path.join(nervusenv.sets_dir, date_name)
    os.makedirs(save_date_dir, exist_ok=True)
    os.makedirs(os.path.join(save_date_dir, nervusenv.weight_dir), exist_ok=True)


    # Training
    logger.info('Training started...')
    logger.info(f"train_data = {len(train_loader.dataset)}")
    logger.info(f"  val_data = {len(val_loader.dataset)}")

    best_weight, val_best_loss, val_best_epoch, loss_acc_dict, val_best_loss_label_wise_dict, val_best_epoch_label_wise_dict, loss_acc_label_wise_dict = execute(save_date_dir, save_weight, best_weight, val_best_loss, val_best_epoch, loss_acc_dict, val_best_loss_label_wise_dict, val_best_epoch_label_wise_dict, loss_acc_label_wise_dict, label_list)

    logger.info('Training finished!')

    save_result(save_date_dir, best_weight, val_best_loss, val_best_epoch, loss_acc_dict, val_best_loss_label_wise_dict, val_best_epoch_label_wise_dict, loss_acc_label_wise_dict)

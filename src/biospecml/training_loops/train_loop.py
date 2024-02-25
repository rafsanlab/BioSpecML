import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score, accuracy_score
import numpy as np
import json
import os

# --------------- helper functions ---------------

def calc_metric_prediction(inputs, outputs, metrics_list=['accuracy', 'f1'], f1_average='macro'):
    """
    Evaluate the performance of a model for classification tasks using various metrics.

    Args:
    - inputs (torch.Tensor): Original input labels (ground truth).
    - outputs (torch.Tensor): Predicted output labels.
    - metric (str): Metric to compute. Options: 'accuracy', 'f1', 'all' (default).
    - threshold (float): Threshold for binary classification (default: 0.5).

    Returns:
    - result (dict): Computed metric value or a dictionary containing different evaluation metrics.
    """
    metrics = {}

    if not isinstance(inputs, np.ndarray) or not isinstance(outputs, np.ndarray):
        raise Exception('Inputs/outputs must be numpy array for predictions.')

    for metric in metrics_list:

        if metric not in ['accuracy', 'f1']:
            raise ValueError(f"Invalid metric. Choose from 'accuracy' or/and 'f1'.")

        if metric == 'f1':
            f1 = f1_score(inputs, outputs, average=f1_average)
            metrics['f1'] = f1

        if metric == 'accuracy':
            accuracy = accuracy_score(inputs, outputs)
            metrics['accuracy'] = accuracy

    return metrics


# --------------- running loop ---------------

def train_model(model, data_loader, device, num_epochs, criterion, optimizer,
              savedir=None, f1_average:str='macro', validation_mode:bool=False,
              one_epoch_mode:bool=False, metrics_list:list=None,
              ):
    """
    A basic running loop.

    """


    # if this function is in another running loop, set one_epoch_mode to True
    # so regardless any epoch number, will only run for one
    # but we save the given epoch for stats
    if one_epoch_mode:
        ori_epochs = num_epochs
        num_epochs = 1

    # ----- prepare metrics dictionary -----

    if metrics_list == None:
        metrics_list = ['accuracy', 'f1']
    metrics = {key: [] for key in metrics_list}
    metrics['loss'], metrics['epoch'] = [], []

    for epoch in range(num_epochs):

        epoch = ori_epochs if one_epoch_mode else epoch+1
        epoch_metrics = {key: 0.0 for key in metrics.keys()}        
        epoch_metrics['epoch'] = ori_epochs if one_epoch_mode else epoch

        loop_count = 0 # this track batch number (more robust than using batch_num)

        for data in data_loader:

            loop_count += 1
            inputs, targets = data[0], data[1]

            # check and convert target to long() dtype
            if targets.dtype != torch.long:
                targets = targets.long()

            # ----- forward/backward pass -----

            inputs, targets = inputs.to(device), targets.to(device)
            model.to(device)

            if validation_mode:
                model.eval()
                with torch.no_grad():
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
            else:
                model.train()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # ----- get predictions -----

            if device != 'cpu':
                outputs = outputs.cpu()
                targets = targets.cpu()

            preds = torch.argmax(outputs, dim=1).numpy()
            targets = targets.numpy()

            # ----- get metrics -----

            batch_metrics = calc_metric_prediction(preds, targets, metrics_list, f1_average)
            
            for key in batch_metrics.keys():
                epoch_metrics[key] += batch_metrics[key]
            epoch_metrics['loss'] += loss.item()

        for key in epoch_metrics.keys():
            if key != 'epoch': # escape 'epoch' value because we doing division
                epoch_metrics[key] /= loop_count


        # ----- print, fname and append -----

        # set stats fname
        if validation_mode:
            text1 = 'VALIDATE'
            stat_fname = 'stats_val.json'
        else:
            text1 = 'TRAINING '
            stat_fname = 'stats_train.json'

        # print and append metrics
        print(f'{text1} Epoch {epoch:03d}', end=" - ")
        for key, value in epoch_metrics.items():
            metrics[key].append(value)
            if key!= 'epoch':
                print(f"{key} : {value:.6f}", end=" | ")
        print()

        # save metrics every epoch to be safe
        if savedir != None:
            dir_metrics = os.path.join(savedir, stat_fname)
        else:
            dir_metrics = os.path.join(os.getcwd(), stat_fname)
        with open(dir_metrics, 'w') as json_file:
            json.dump(metrics, json_file, indent=4)

    return model, metrics

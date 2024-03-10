# from sklearn.metrics import mean_squared_error, mean_absolute_error
# from skimage.metrics import structural_similarity as ssim
# from skimage.metrics import peak_signal_noise_ratio as psnr
# from sklearn.metrics import f1_score, accuracy_score
from ..ml.metrics_functions import calc_metric_prediction, calc_metric_similarity
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
import os

# --------------- helper functions ---------------

# def calc_metric_prediction(inputs, outputs, metrics_list=['accuracy', 'f1'], f1_average='macro'):
#     """
#     Evaluate the performance of a model for classification tasks using various metrics.

#     Args:
#     - inputs (torch.Tensor): Original input labels (ground truth).
#     - outputs (torch.Tensor): Predicted output labels.
#     - metric (str): Metric to compute. Options: 'accuracy', 'f1', 'all' (default).
#     - threshold (float): Threshold for binary classification (default: 0.5).

#     Returns:
#     - result (dict): Computed metric value or a dictionary containing different evaluation metrics.
#     """
#     metrics = {}

#     if not isinstance(inputs, np.ndarray) or not isinstance(outputs, np.ndarray):
#         raise Exception('Inputs/outputs must be numpy array for predictions.')

#     for metric in metrics_list:

#         if metric not in ['accuracy', 'f1']:
#             raise ValueError(f"Invalid metric. Choose from 'accuracy' or/and 'f1'.")

#         if metric == 'f1':
#             f1 = f1_score(inputs, outputs, average=f1_average)
#             metrics['f1'] = f1

#         if metric == 'accuracy':
#             accuracy = accuracy_score(inputs, outputs)
#             metrics['accuracy'] = accuracy

#     return metrics


# def calc_metric_similarity(inputs, outputs, metrics_list=['SSIM']):
#     """
#     Evaluate the performance of a recontructive model using various metrics.

#     Args:
#     - inputs (np.ndarray): Original input images.
#     - outputs (np.ndarray): Reconstructed output images.
#     - metric (str): Metric to compute. Options: 'MSE', 'BCE', 'MAE', 'SSIM', 'PSNR'.

#     Returns:
#     - metrics (dict): Computed metric value or a dictionary containing different evaluation metrics.

#     Example:
#     >>> arr1, arr2 = np.random.rand(1, 3, 16, 16), np.random.rand(1, 3, 16, 16)
#     >>> metrics = calc_metric_similarity(arr1, arr2, metrics_list=['SSIM'])

#     """
#     metrics = {}
#     metrics_list_ref = ['MSE', 'MAE', 'SSIM', 'PSNR']

#     for metric in metrics_list:

#         # check metrics
#         if metric not in metrics_list_ref:
#             raise ValueError(f"Invalid metric. Choose among {metrics_list_ref}")

#         # Mean Squared Error (MSE)
#         if metric=='MSE':
#             metrics['MSE'] = mean_squared_error(inputs, outputs)

#         # Mean Absolute Error (MAE)
#         if metric=='MAE':
#             metrics['MAE'] = mean_absolute_error(inputs, outputs)

#         # Structural Similarity Index (SSIM)
#         if metric=='SSIM':
#             multichannel = False if len(inputs.shape) == 2 else True
#             ssim_value = ssim(inputs, outputs, multichannel=multichannel)
#             metrics['SSIM'] = np.mean(ssim_value)

#         # Peak Signal-to-Noise Ratio (PSNR)
#         if metric=='PSNR':
#             metrics['PSNR'] = psnr(inputs, outputs)

#     return metrics


# --------------- running loop ---------------

# --------------- running loop ---------------

def train_model(model, data_loader, device, num_epochs, criterion, optimizer,
              running_type:str='prediction', verbose:bool=True,
              savedir:str=None, f1_average:str='macro', validation_mode:bool=False,
              one_epoch_mode:bool=False, metrics_list:list=None,
              ):
    """
    A basic running loop.

    """
    running_types = ['prediction', 'similarity']
    if running_type not in running_types:
        raise Exception(f'Choose *running_type : {running_types}')
    
    # if this function is in another running loop, set one_epoch_mode to True
    # so regardless any epoch number, will only run for one
    # but we save the given epoch for stats
    if one_epoch_mode:
        ori_epochs = num_epochs
        num_epochs = 1

    # ----- prepare metrics dictionary -----

    if metrics_list == None:
        raise Exception('Please provide metric list.')
    metrics = {key: [] for key in metrics_list}
    metrics['loss'], metrics['epochs'] = [], []

    for epoch in range(num_epochs):

        epoch = ori_epochs if one_epoch_mode else epoch+1
        epoch_metrics = {key: 0.0 for key in metrics.keys()}        
        epoch_metrics['epochs'] = ori_epochs if one_epoch_mode else epoch

        loop_count = 0 # this track batch number (more robust than using batch_num)

        for data in data_loader:

            loop_count += 1
            inputs, targets = data[0], data[1]

            # check and convert target to long() dtype
            if running_type=='prediction':
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

            if device != 'cpu':
                outputs = outputs.cpu()
                targets = targets.cpu()

            # ----- get metrics -----
                
            # get prediction metrics
            if running_type=='prediction':
                preds = torch.argmax(outputs, dim=1).numpy()
                targets = targets.numpy()
                batch_metrics = calc_metric_prediction(preds, targets, metrics_list, f1_average)
            
            # get similarity metrics
            if running_type=='similarity':
                outputs, targets = outputs.detach().numpy(), targets.detach().numpy()
                batch_metrics = calc_metric_similarity(outputs, targets, metrics_list)
  
            for key in batch_metrics.keys():
                epoch_metrics[key] += batch_metrics[key]
            epoch_metrics['loss'] += loss.item()

        for key in epoch_metrics.keys():
            if key != 'epochs': # escape 'epoch' value because we doing division
                epoch_metrics[key] /= loop_count

        # ----- print, stat fname and append metrics -----

        # set stats fname
        if validation_mode:
            text1 = 'VALIDATE '
            stat_fname = 'stats_val.json'
        else:
            text1 = 'TRAINING '
            stat_fname = 'stats_train.json'

        # print and append metrics
        # print(f'{text1} Epoch {epoch:03d}', end=" - ")
        for key, value in epoch_metrics.items():
            metrics[key].append(value)
            # if key!= 'epochs':
                # print(f"{key} : {value:.6f}", end=" | ")
        # print()

        # print some stats
        if verbose:
            print(f'{text1} Epoch {epoch:03d}', end=" - ")
            for key, value in epoch_metrics.items():
                if key!= 'epochs':
                    print(f"{key} : {value:.6f}", end=" | ")
            print()
                
        # condition to save metrics, save every epoch to be safe
        if savedir != None:
            dir_metrics = os.path.join(savedir, stat_fname)
            with open(dir_metrics, 'w') as json_file:
                json.dump(metrics, json_file, indent=4)

    return model, metrics


def train_val_loop(model, device, num_epochs, criterion, optimizer,
                   train_loader, test_loader=None, trained_num_epochs:int=None,
                   running_type:str='predictions', verbose:bool=True,
                   f1_average:str='macro', metrics_list:list=['f1', 'accuracy'],
                   savedir:str=None, epoch_save_checkpoints:list=[],
                   save_model:bool=True,
                   ):
    """
    Example of use:
    >>> learning_rate , weight_decay = 0.001, 0
    >>> model, main_metrics = train_val_loop(
    >>>     model = model,
    >>>     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    >>>     num_epochs = 3,
    >>>     criterion = nn.CrossEntropyLoss(),
    >>>     optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay),
    >>>     train_loader = train_loader,
    >>>     test_loader = test_loader,
    >>>     running_type = 'prediction',
    >>>     f1_average = 'macro',
    >>>     metrics_list = ['f1', 'accuracy'],
    >>>     savedir = os.getcwd(),
    >>>     epoch_save_checkpoints = [2,3]
    >>> )
    """
    start_epoch = trained_num_epochs+1 if isinstance(trained_num_epochs, int) else 1
    main_metrics = {}
    main_metrics['epochs'] = []
    savedir = os.getcwd() if savedir is None or savedir == '' else savedir
    traindir = os.path.join(savedir, 'training')
    os.makedirs(traindir, exist_ok=True)

    for epoch in range(start_epoch, num_epochs+1, 1):

        # ----- run training and validation -----
        
        container_metrics = []
        
        if train_loader != None:
            model, train_metrics = train_model(
                model, train_loader, device, epoch, criterion, optimizer,
                running_type = running_type, f1_average = f1_average,
                validation_mode = False,
                one_epoch_mode = True, 
                metrics_list = metrics_list,
                verbose = False,
                )
            container_metrics.append(("train", train_metrics))

        if test_loader != None:
            model, test_metrics = train_model(
                model, test_loader, device, epoch, criterion, optimizer,
                running_type = running_type, f1_average = f1_average,
                validation_mode = True,
                one_epoch_mode = True, 
                metrics_list = metrics_list,
                verbose = False,
                )
            container_metrics.append(("val.", test_metrics))
        
        # ----- collect metrics -----

        main_metrics['epochs'].append(epoch)
        for phase, metrics in container_metrics:
            for k, v in metrics.items():
                new_k = f'{phase} {k}' # combine phase to dict's key
                if new_k not in main_metrics:
                    main_metrics[new_k] = v
                else:
                    main_metrics[new_k].extend(v)


        # print some metrics stats
        if verbose:
            print(f'Epoch {epoch:03d}', end=" : ")
            for phase, metrics in container_metrics:
                print(f'|| {phase.upper()}', end=' |')
                for key, value in metrics.items():
                    if 'epochs' not in key:
                        print(f"| {key} : {value[-1]:.6f}", end=" ")
            print()

        # save metrics
        stat_fname_path = os.path.join(traindir, f'stats_e{num_epochs}.json')
        with open(stat_fname_path, 'w') as json_file:
                json.dump(main_metrics, json_file, indent=4)

        # ----- option to save checkpoints -----

        if len(epoch_save_checkpoints)>0 and epoch in epoch_save_checkpoints:
            checkpoint_path = os.path.join(savedir, 'checkpoints')
            checkpoint_save_path = os.path.join(checkpoint_path, 'checkpoint_e'+str(epoch)+'.pth')
            checkpoint_stat_path = os.path.join(checkpoint_path, 'checkpoint_e'+str(epoch)+'_stats.json')
            if not os.path.exists(checkpoint_path): os.makedirs(checkpoint_path) 
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epochs': epoch,
                }
            torch.save(checkpoint, checkpoint_save_path)
            with open(checkpoint_stat_path, 'w') as json_file:
                json.dump(main_metrics, json_file, indent=4)
            print(f'Saved checkpoint at epoch {epoch}.')
 
    # ----- option to save model -----

    if save_model:
        model_path = os.path.join(traindir, f'model_e{epoch}.pth')
        torch.save(model, model_path)
        print(f'Saved model at epoch {epoch}.')

    return model, main_metrics

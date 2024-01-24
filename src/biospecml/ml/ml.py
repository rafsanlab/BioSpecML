import numpy as np
import pandas as pd
from scipy import ndimage
import torch
import random
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
# import tqdm

import torch.nn as nn
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from sklearn.metrics import accuracy_score, f1_score
import time
import json
import os

def create_patches(image_array, patch_size=(32, 32), step=(16, 16)):
    """
    Create patches from an image array.
    [!] Expects channel at the back.

    Parameters:
    - image_array: numpy array of an image; (256, 256, 3).
    - patch_size: size of pathces.
    - step: x and y step to patch.

    Returns:
    - patches: List of patches arrays.

    """ 
    patches = []
    height, width = image_array.shape[:2]

    for y in range(0, height - patch_size[0] + 1, step[0]):
        for x in range(0, width - patch_size[1] + 1, step[1]):
            patch = image_array[y:y + patch_size[0], x:x + patch_size[1]]
            patches.append(patch)

    return patches

def plot_patches(patches, num_cols=5, title_size=10, fig_size=(12, 8), show_plot=True, fname=None):
    """
    Plot the patches in a grid.

    Parameters:
    - patches: List of numpy arrays representing patches.
    - num_cols: Number of columns in the grid.
    - fig_size: Tuple specifying the figure size (width, height).
    - show_plot: Boolean indicating whether to display the plot.
    - fname: File name to save the plot. If None, the plot is not saved.

    Returns:
    - None (displays the plot or saves it to a file).
    """

    num_patches = len(patches)
    num_rows = (num_patches + num_cols - 1) // num_cols

    plt.figure(figsize=fig_size)

    for i, patch in enumerate(patches, start=1):
        plt.subplot(num_rows, num_cols, i)
        plt.imshow(patch, cmap='gray')
        plt.title(f'Patch {i}', fontsize=title_size)
        plt.axis('off')

    plt.tight_layout()

    if fname is not None:
        plt.savefig(fname)

    if show_plot:
        plt.show()
    else:
        plt.close()


def rotate_and_random_flip(images=[], pair_mode=False, rotate=True, rotate_method='numpy',
                           rotation_range=180, minimum_angle=30, rotation_angle=90, mode='nearest',
                           flip_horizontal=True, flip_vertical=False):
    """
    Arguments:
        - expect images in an array, i.e: [img1, img2, ..]
        - or in pair_mode, expect paired images, i.e: [[img1, img2, ..], ..]
        - expect (w x h x c) image format
        - rotate_method='numpy' will rotate only 90 deg step, set in rotation_angle i.e: 90, 180..
        - rotate_method='scipy' will use rotation_range, minimum_angle and mode
    Returns:
        augmented_image

    """
    augmented_images = []
    for image in images:

        if rotate:

            # Random rotation using scipy
            if rotate_method=='scipy':
                angle = random.randint(0, rotation_range + minimum_angle)
                if pair_mode:
                    augmented_image = []
                    for pair in image:
                        augmented_pair = ndimage.rotate(pair, angle, axes=(0, 1), reshape=True, mode=mode)
                        augmented_image.append(augmented_pair)
                else:
                    augmented_image = ndimage.rotate(image, angle, axes=(0, 1), reshape=True, mode=mode)

            # 90 rotation using numpy
            elif rotate_method=='numpy':
                if pair_mode:
                    augmented_image = []
                    for pair in image:
                        augmented_pair = np.rot90(pair, k=rotation_angle // 90)
                        augmented_image.append(augmented_pair)
                else:
                    augmented_image = np.rot90(image, k=rotation_angle // 90)
        else:
            augmented_image = image

        if flip_horizontal or flip_vertical:

            # track if image is at least flipped
            flip_signal = 0

            while flip_signal==0:

                # Random horizontal flip
                if flip_horizontal and random.choice([True, False]):
                    if pair_mode:
                        temp_ = []
                        for pair in augmented_image:
                            temp_aug = np.flip(pair, axis=1)
                            temp_.append(temp_aug)
                        augmented_image = temp_
                    else:
                        augmented_image = np.flip(augmented_image, axis=1)
                    flip_signal += 1

                # Random vertical flip
                if flip_horizontal and random.choice([True, False]):
                    if pair_mode:
                        temp_ = []
                        for pair in augmented_image:
                            temp_aug = np.flip(pair, axis=1)
                            temp_.append(temp_aug)
                        augmented_image = temp_
                    else:
                        augmented_image = np.flip(augmented_image, axis=0)
                    flip_signal += 1

        augmented_images.append(augmented_image)
    return augmented_images


def rotate_and_flip(images=[], pair_mode=False, rotate=True, rotate_method='numpy',
                           rotation_range=180, minimum_angle=30, rotation_angle=90, mode='nearest',
                           flip_horizontal=True, flip_vertical=False):
    """
    Arguments:
        - expect images in an array, i.e: [img1, img2, ..]
        - or in pair_mode, expect paired images, i.e: [[img1, img2, ..], ..]
        - expect (w x h x c) image format
        - rotate_method='numpy' will rotate only 90 deg step, set in rotation_angle i.e: 90, 180..
        - rotate_method='scipy' will use rotation_range, minimum_angle and mode
    Returns:
        augmented_image

    """
    augmented_images = []
    for image in images:

        if rotate:

            # Random rotation using scipy
            if rotate_method=='scipy':
                angle = random.randint(0, rotation_range + minimum_angle)
                if pair_mode:
                    augmented_image = []
                    for pair in image:
                        augmented_pair = ndimage.rotate(pair, angle, axes=(0, 1), reshape=True, mode=mode)
                        augmented_image.append(augmented_pair)
                else:
                    augmented_image = ndimage.rotate(image, angle, axes=(0, 1), reshape=True, mode=mode)

            # 90 rotation using numpy
            elif rotate_method=='numpy':
                if pair_mode:
                    augmented_image = []
                    for pair in image:
                        augmented_pair = np.rot90(pair, k=rotation_angle // 90)
                        augmented_image.append(augmented_pair)
                else:
                    augmented_image = np.rot90(image, k=rotation_angle // 90)
        else:
            augmented_image = image

        if flip_horizontal:

            if pair_mode:
                temp_ = []
                for pair in augmented_image:
                    temp_aug = np.flip(pair, axis=1)
                    temp_.append(temp_aug)
                augmented_image = temp_
            else:
                augmented_image = np.flip(augmented_image, axis=1)
    
        if flip_vertical:
            
            if pair_mode:
                temp_ = []
                for pair in augmented_image:
                    temp_aug = np.flip(pair, axis=1)
                    temp_.append(temp_aug)
                augmented_image = temp_
            else:
                augmented_image = np.flip(augmented_image, axis=0)

        augmented_images.append(augmented_image)

    return augmented_images


def calc_DatasetMeanStd(loader, channels, data_type=torch.float32, 
                        data_position=None, replace_zeros=False,
                        replace_value=1.4013e-45):
    """
    Calculate mean and std of the channels in each image in the data loader,
    and average them, expect approximation of the values due to averaging because
    stacking all ftir tensors tend to be huge to load in memory.

    Args:
        loader(torch.utils.data.DataLoader): accept data loader that output single or double data pair
        channels(int): number of channels of the image
        data_type(torch.dtype): torch data type to hold the mean/std value
        data_position(int): position of data output from data loader (0 or 1)
        replace_zeros(bool): condition to replace zeros in mean/std value
        replace_value(float): the default is the minimum value in float32
    Returns:
        mean(float): mean of all data in data loader
        std(float): std of all data in data loader
    """

    #--- declare variables ---
    total_mean = torch.zeros(channels, dtype=data_type)
    total_std = torch.zeros(channels, dtype=data_type)
    count = 0

    #--- condition of the data loader outputs ---
    """ in the case of data loader that output two set of data (i.e: image, label)
        the following condition allow user to select where is the position (0 or 1) of the
        data for calculation (image or label).
    """
    for batch in loader:
        if len(batch) == 1:
            # the data loader only output 1 data
            img = batch
        elif len(batch) == 2:
            # the data loader output 2 data
            img = batch[data_position]
        else:
            # the data loader output >= 2 data
            raise Exception('Only support either 0 or 1 data position.')

        #--- calculate mean/std ---
        mean = img.view(channels, -1).mean(dim=1)
        std = img.view(channels, -1).std(dim=1)
        total_mean += mean; total_std += std; count += 1

    #--- averaging mean/std ---
    ave_mean, ave_std = total_mean/count, total_std/count

    #--- replace zero condition ---
    """ important to replace zeros if data will be use for dataset normalisation 
        because division error will occur if there's zeros values in std/mean.
    """
    if replace_zeros==True:
        data = [ave_mean, ave_std] # compile averages into a list
        for ave in data: # loop through averages
            zero_indices = (ave==0) # find any zeros in the mean/std
            if (zero_indices==True).any()==True:
                print('Replacing zero(s)..')
                ave[zero_indices] = replace_value # replacement

    return ave_mean, ave_std


def calc_DatasetMeanStd_byDf(df, replace_zeros=True, replace_nans=True,
                             replace_value=1.4013e-45):
    """
    Calculate the mean and standard deviation of numeric columns in a DataFrame.
    [*] Expects features as columns, readings are rows.

    Args:
        df (pd.DataFrame): Input DataFrame.
        replace_zeros (bool, optional): Replace zeros with `replace_value`. Default is True.
        replace_nans (bool, optional): Replace NaNs with `replace_value`. Default is True.
        replace_value (float, optional): Value used for replacement of zeros and NaNs.

    Returns:
        Tuple[pd.Series, pd.Series]: Mean and standard deviation values for each numeric column.

    """
    mean_values = df.mean(numeric_only=True)
    std_values = df.std(numeric_only=True)
    if replace_zeros==True:
        mean_values.replace(0, replace_value, inplace=True)
        std_values.replace(0, replace_value, inplace=True)
    if replace_nans==True:
        mean_values.fillna(replace_value, inplace=True)
        std_values.fillna(replace_value, inplace=True)
    return mean_values, std_values


def calc_DatasetNorm(df, drop_cols:list=None, save_cols:list=None, norm_axis=1, replace_zeros=False,
                     replace_nans=False, replace_value=1.4013e-45, test_mode=False, n_counter=2):
    """ In case of dataset too big and loading a single df is a issue,
    read df with <chunksize> parameter and feed that df into this function. 
    [*] Expects wavenumber and metadata as columns, readings as rows
    Args:
        - df

    """
    
    # df_norm = pd.DataFrame()
    df_norm = []
    counter = 0

    for chunk in df:

        saved_cols = []

        #--- drop columns ---
        if save_cols != None:
            for col in save_cols:
                saved_cols.append(chunk[col])

        #--- drop columns ---
        if drop_cols != None:
            for col in drop_cols:
                chunk.drop([col], axis=1, inplace=True)
        
        #--- chunck normalisation ---
        normalized_chunk = chunk.apply(lambda x: x / np.linalg.norm(x), axis=norm_axis)

        #--- replace NaNs ---
        if replace_nans!=False:
            normalized_chunk.fillna(replace_value, inplace=True)

        #--- replace zeros ---
        if replace_zeros!=False:
            normalized_chunk.replace(0.0, replace_value, inplace=True)

        for i, col in enumerate(save_cols):
            normalized_chunk[col] = saved_cols[i]

        #--- combine normalised chunk to main df ---
        # df_norm = pd.concat([df_norm, normalized_chunk], axis=norm_axis, ignore_index=True)
        df_norm.append(normalized_chunk)

        #--- condition of <test_mode>    
        if test_mode!=False:
            counter+=1
            if counter!=n_counter:
                continue
            elif counter==n_counter:
                break

    df_normed = pd.concat(df_norm, ignore_index=True)
    
    return df_normed

# def calc_DatasetMeanStd(loader, channels, data_type=torch.float64, data_position=None):
#     """
#     Calculate mean and std of the channels in each image in the data loader,
#     and average them, expect approximation of the values due to averaging because
#     stacking all ftir tensors tend to be huge to load in memory.

#     Args:
#         loader(torch.utils.data.DataLoader): accept data loader that output single or double data pair
#         channels(int): number of channels of the image
#         data_position(int): position of data output from data loader (0 or 1)
#     Returns:
#         mean(float): mean of all data in data loader
#         std(float): std of all data in data loader
#     """

#     # __declare variables__
#     total_mean = torch.zeros(channels, dtype=data_type)
#     total_std = torch.zeros(channels, dtype=data_type)
#     count = 0

#     """ in the case of data loader that output two set of data (i.e: image, label)
#         the following condition allow user to select where is the position (0 or 1) of the
#         data for calculation (image or label).
#     """
#     for batch in loader:
#         if len(batch) == 1:
#             # the data loader only output 1 data
#             # print('Loader output 1 data')
#             img = batch
#         elif len(batch) == 2:
#             # the data loader output 2 data
#             # print('Loader output 2 data, getting data_position.')
#             img = batch[data_position]
#         else:
#             # the data loader output >= 2 data
#             raise Exception('Only support either 0 or 1 data position.')

#         # __calculations__
#         mean = img.view(channels, -1).mean(dim=1)
#         std = img.view(channels, -1).std(dim=1)
#         total_mean += mean; total_std += std; count += 1

#     # calculate averages  
#     ave_mean, ave_std = total_mean/count, total_std/count
#     return ave_mean, ave_std


def save_tensor_to_file(tensor, fname=None, delimiter='\t'):
    """
    convert tensor to file (default:.tsv) file.
    <fname> is the path the save dir with file name.

    """
    numpy_array = tensor.numpy()
    if fname!=None:
        np.savetxt(fname, numpy_array, delimiter=delimiter)
        print(f"Tensor saved to '{fname}'.")
    else:
        return numpy_array


def load_tensor_from_file(file_path, dtype=np.float32, delimiter='\t'):
    """
    convert file (default:.tsv) to tensor.
    
    """
    numpy_array = np.loadtxt(file_path, delimiter=delimiter, dtype=dtype)
    tensor = torch.from_numpy(numpy_array)
    return tensor


def upsampling_via_smote(dfX, labels, random_state=42, sampling_strategy='auto'):
    """
    SMOTE on X matrix of a df.

    """
    dfX.columns = dfX.columns.astype(str)
    y = labels
    smote = SMOTE(sampling_strategy=sampling_strategy, random_state=random_state)
    X_resampled, y_resampled = smote.fit_resample(dfX, y)
    return X_resampled, y_resampled


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


def calc_metric_reconstruction(inputs, outputs, metrics_list=['MSE', 'BCE', 'MAE', 'SSIM', 'PSNR']):
    """
    Evaluate the performance of a recontructive model using various metrics.

    Args:
        - inputs (torch.Tensor): Original input images.
        - outputs (torch.Tensor): Reconstructed output images.
        - metric (str): Metric to compute. Options: 'MSE', 'BCE', 'MAE', 'SSIM', 'PSNR'.

    Returns:
        - metrics (dict): Computed metric value or a dictionary containing different evaluation metrics.

    Example:
        >>> tnsr1, tnsr2 = torch.rand(1, 3, 16, 16), torch.rand(1, 3, 16, 16)
        >>> metrics = calc_metric_reconstruction(tnsr1, tnsr2)

    """
    metrics = {}

    for metric in metrics_list:
        if metric not in ['MSE', 'BCE', 'MAE', 'SSIM', 'PSNR']:
            raise ValueError(f"Invalid metric. Choose among 'MSE', 'BCE', 'MAE', 'SSIM', 'PSNR'")

        if metric=='MSE':
            # Mean Squared Error (MSE)
            mse_loss = nn.MSELoss()
            metrics['MSE'] = mse_loss(inputs, outputs).item()

        if metric=='BCE':
            # Binary Cross-Entropy Loss (BCE)
            bce_loss = nn.BCELoss()
            metrics['BCE'] = bce_loss(outputs, inputs).item()

        if metric=='MAE':
            # Mean Absolute Error (MAE)
            mae_loss = nn.L1Loss()
            metrics['MAE'] = mae_loss(inputs, outputs).item()

        if metric=='SSIM':

            # check inputs and outputs
            x, y = len(inputs.shape), len(outputs.shape)
            if x != 4 or y != 4:
                raise Exception('SSIM need batched input.')

            # Structural Similarity Index (SSI)
            batch_num = inputs.shape[0]
            channel_num = inputs.shape[1]
            ssim_values = np.zeros(batch_num)
            arr1, arr2 = inputs.squeeze().cpu().numpy(), outputs.squeeze().cpu().numpy()

            # Calculate SSIM for each channel separately
            for i in range(batch_num):
                for channel in range(channel_num):
                    ssim_values[i] += ssim(arr1[i, channel], arr2[i, channel])

            ssim_values /= channel_num  # Average SSIM across channels
            average_ssim = np.mean(ssim_values)  # Average SSIM across the batch
            metrics['SSIM'] = average_ssim

        if metric=='PSNR':
            # Peak Signal-to-Noise Ratio (PSNR)
            psnr_value = psnr(inputs.squeeze().cpu().numpy(), outputs.squeeze().cpu().numpy())
            metrics['PSNR'] = psnr_value

    return metrics


def model_train(model, data_loader, criterion, optimizer, send_to_device=True, mode='prediction',
                metrics_list=['accuracy', 'f1'], f1_average='macro'):
    """
    TRAINING FUNCTION FOR MODEL

    """
    if send_to_device:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
    model.train()

    total_loss = 0.0
    metrics = {}
    batch_metrics = {}

    # for inputs_batch, labels_batch in data_loader:
    for data in data_loader:

        if send_to_device:
            inputs, labels = data[0].to(device), data[1].to(device)

        optimizer.zero_grad()
        outputs = model(inputs)

        if mode == 'prediction':
            loss = criterion(outputs, labels.long())
        elif mode == 'reconstruction':
            loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if mode == 'prediction':

            _, preds = torch.max(outputs, 1)
            preds, labels = preds.cpu().numpy(), labels.cpu().numpy()
            batch_metrics = calc_metric_prediction(labels, preds, metrics_list=metrics_list, f1_average=f1_average)
            outputs = preds.copy()

            for key in batch_metrics:
                if key not in metrics.keys():
                    metrics[key] = 0.0
                metrics[key] += batch_metrics[key]

        elif mode == 'reconstruction':

            outputs, labels = outputs.detach(), labels.detach()
            batch_metrics = calc_metric_reconstruction(labels, outputs, metrics_list=metrics_list)

            for key in batch_metrics:
                if key not in metrics.keys():
                    metrics[key] = 0.0
                metrics[key] += batch_metrics[key]

    for key in batch_metrics.keys():
        metrics[key] /= len(data_loader)

    loss = total_loss / len(data_loader)
    metrics['loss'] = loss

    return metrics


def model_test(model, data_loader, mode='prediction', send_to_device=True,
                metrics_list=['accuracy', 'f1'], f1_average='macro'):
    """
    TESTING FUNCTION FOR MODEL

    """
    if send_to_device:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
    model.eval()

    metrics = {}
    batch_metrics = {}

    with torch.no_grad():
        # for inputs_batch, labels_batch in data_loader:
        for data in data_loader:
            if send_to_device:
                inputs, labels = data[0].to(device), data[1].to(device)

            outputs = model(inputs)

            if mode == 'prediction':

                _, outputs = torch.max(outputs, 1)
                outputs, labels = outputs.cpu().numpy(), labels.cpu().numpy()
                if metrics_list is not None:
                    batch_metrics = calc_metric_prediction(labels, outputs, metrics_list=metrics_list, f1_average=f1_average)

                    for key in batch_metrics:
                        if key not in metrics.keys():
                            metrics[key] = 0.0
                        metrics[key] += batch_metrics[key]

            elif mode == 'reconstruction':

                outputs, labels = outputs.detach(), labels.detach()

                if metrics_list is not None:
                    batch_metrics = calc_metric_reconstruction(labels, outputs, metrics_list=metrics_list)

                    for key in batch_metrics:
                        if key not in metrics.keys():
                            metrics[key] = 0.0
                        metrics[key] += batch_metrics[key]

        for key in batch_metrics.keys():
            metrics[key] /= len(data_loader)

    return metrics, outputs, labels


def run_training_testing(model, train_loader, test_loader, num_epochs, criterion,
                         optimizer, metrics_list, verbose=True,
                         mode='reconstruction', run_training=True, run_testing=False,
                         epoch_save_checkpoint=[], project_dir=None, save_model=False,
                         trained_num_epochs=None, f1_average='macro'):

    main_metrics = {} # to compile all the epoch metrics
    
    # check the main directory to store all files
    if project_dir==None:
        project_dir = os.getcwd()
    
    # --------------- start epoch ---------------
    
    # for epoch in tqdm(range(1, num_epochs+1, 1)):
    for epoch in range(1, num_epochs+1, 1):
        
        # add epochs (for model loaded from checkpoints)
        if isinstance(trained_num_epochs, int):
            epoch += trained_num_epochs
        
        # housekeeping
        start_time = time.time() 
        epoch_metrics = {}
        epoch_metrics['epochs'] = epoch
        
        # --------------- training ---------------
        if run_training:
            training_metrics = model_train(model, train_loader, criterion, optimizer, mode=mode, metrics_list=metrics_list, f1_average=f1_average)
            
            # save metrics
            for key in training_metrics.keys():
                training_key = 'training ' + key
                if training_key not in epoch_metrics.keys():
                    epoch_metrics[training_key] = None
                epoch_metrics[training_key] = training_metrics[key]
            
            # save time per epoch
            epoch_time = time.time() - start_time  # Calculate the time taken for the epoch
            epoch_metrics['training epoch time'] = epoch_time  # Add epoch time to metrics
        
        # --------------- testing ---------------
        if run_testing:
            testing_metrics, outputs, labels = model_test(model, test_loader, mode=mode, metrics_list=metrics_list, f1_average=f1_average)
            
            # save metrics
            for key in testing_metrics.keys():
                testing_key = 'testing ' + key
                if testing_key not in epoch_metrics.keys():
                    epoch_metrics[testing_key] = None
                epoch_metrics[testing_key] = testing_metrics[key]
            
            # save time per epoch
            epoch_time = time.time() - start_time  # Calculate the time taken for the epoch
            epoch_metrics['training epoch time'] = epoch_time  # Add epoch time to metrics

        
        # --------------- epoch etcs. ---------------
        
        # epoch statistics
        for key, value in epoch_metrics.items():
            if key not in main_metrics:
                main_metrics[key] = [value]
            else:
                main_metrics[key].append(value)
        
        # verbose mode to show stat during training
        if verbose:
            print(f"----- Epoch {epoch} -----")
            for metric in epoch_metrics:
                print(f"{metric} \t: {main_metrics[metric][-1]}")
            print()
        
        # save stat data
        stat_fname_path = os.path.join(project_dir, 'stats.json')
        with open(stat_fname_path, 'w') as json_file:
                json.dump(main_metrics, json_file, indent=4)
    
        # --------------- save model checkpoint ---------------
        
        if len(epoch_save_checkpoint)>0:
        # if isinstance(epoch_save_checkpoint, int):
            checkpoint_path = os.path.join(project_dir, 'checkpoints')
            if not os.path.exists(checkpoint_path):
                os.makedirs(checkpoint_path)
            if epoch in epoch_save_checkpoint:
                checkpoint = {
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch,
                }
                print('Saving checkpoints at epoch ', epoch)
                checkpoint_save_path = os.path.join(checkpoint_path, 'checkpoint_e'+str(epoch)+'.pth')
                torch.save(checkpoint, checkpoint_save_path)
                
                checkpoint_stat_fname_path = os.path.join(checkpoint_path, 'checkpoint_e'+str(epoch)+'_stats.json')
                with open(checkpoint_stat_fname_path, 'w') as json_file:
                    json.dump(main_metrics, json_file, indent=4)
 
    # save final model
    if save_model:
        model_path = os.path.join(project_dir, 'model.pth')
        torch.save(model, model_path)
    
    # --------------- returns ---------------
    
    if run_testing:
        return model, main_metrics, outputs, labels
    else:
        return model, main_metrics

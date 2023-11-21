import numpy as np
import pandas as pd
from scipy import ndimage
import torch
import random

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
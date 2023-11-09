import numpy as np
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

def calc_DatasetMeanStd(loader, channels, data_position=None):
    """
    Calculate mean and std of the channels in each image in the data loader,
    and average them, expect approximation of the values due to averaging because
    stacking all ftir tensors tend to be huge to load in memory.

    Args:
        loader(torch.utils.data.DataLoader): accept data loader that output single or double data pair
        channels(int): number of channels of the image
        data_position(int): position of data output from data loader (0 or 1)
    Returns:
        mean(float): mean of all data in data loader
        std(float): std of all data in data loader
    """

    # __declare variables__
    total_mean, total_std = torch.zeros(channels), torch.zeros(channels)
    count = 0

    """ in the case of data loader that output two set of data (i.e: image, label)
        the following condition allow user to select where is the position (0 or 1) of the
        data for calculation (image or label).
    """
    for batch in loader:
        if len(batch) == 1:
            # the data loader only output 1 data
            print('Loader output 1 data')
            img = batch
        elif len(batch) == 2:
            # the data loader output 2 data
            print('Loader output 2 data, getting data_position.')
            img = batch[data_position]
        else:
            # the data loader output >= 2 data
            raise Exception('Only support either 0 or 1 data position.')

        # __calculations__
        mean = img.view(channels, -1).mean(dim=1)
        std = img.view(channels, -1).std(dim=1)
        total_mean += mean; total_std += std; count += 1

    # calculate averages  
    ave_mean, ave_std = total_mean/count, total_std/count
    return ave_mean, ave_std


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
import torch

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
            img = batch
        elif len(batch) == 2:
            # the data loader output 2 data
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
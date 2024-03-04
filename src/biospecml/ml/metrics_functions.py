from sklearn.metrics import mean_squared_error, mean_absolute_error
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from sklearn.metrics import f1_score, accuracy_score
import numpy as np

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


def calc_metric_similarity(inputs, outputs, metrics_list=['SSIM']):
    """
    Evaluate the performance of a recontructive model using various metrics.

    Args:
    - inputs (np.ndarray): Original input images.
    - outputs (np.ndarray): Reconstructed output images.
    - metric (str): Metric to compute. Options: 'MSE', 'BCE', 'MAE', 'SSIM', 'PSNR'.

    Returns:
    - metrics (dict): Computed metric value or a dictionary containing different evaluation metrics.

    Example:
    >>> arr1, arr2 = np.random.rand(1, 3, 16, 16), np.random.rand(1, 3, 16, 16)
    >>> metrics = calc_metric_reconstruction(arr1, arr2, metrics_list=['SSIM'])

    """
    metrics = {}
    metrics_list_ref = ['MSE', 'MAE', 'SSIM', 'PSNR']

    for metric in metrics_list:

        # check metrics
        if metric not in metrics_list_ref:
            raise ValueError(f"Invalid metric. Choose among {metrics_list_ref}")

        # Mean Squared Error (MSE)
        if metric=='MSE':
            metrics['MSE'] = mean_squared_error(inputs, outputs)

        # Mean Absolute Error (MAE)
        if metric=='MAE':
            metrics['MAE'] = mean_absolute_error(inputs, outputs)

        # Structural Similarity Index (SSIM)
        if metric=='SSIM':
            multichannel = False if len(inputs.shape) == 2 else True
            ssim_value = ssim(inputs, outputs, multichannel=multichannel)
            metrics['SSIM'] = np.mean(ssim_value)

        # Peak Signal-to-Noise Ratio (PSNR)
        if metric=='PSNR':
            metrics['PSNR'] = psnr(inputs, outputs)

    return metrics
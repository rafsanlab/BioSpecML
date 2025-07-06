from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from sklearn.metrics import f1_score, accuracy_score
from sklearn.metrics import mean_squared_error, jaccard_score
import numpy as np

# --------------- helper functions ---------------


def calc_metric_prediction(
    inputs,
    outputs,
    metrics_list=["accuracy", "f1"],
    f1_average="macro",
    f1_zero_division='warn',
    labels=None,
):
    """
    Evaluate the performance of a model for classification tasks using various metrics.

    Args:
    - inputs (torch.Tensor): Original input labels (ground truth).
    - outputs (torch.Tensor): Predicted output labels.
    - metrics_list (list): List of metrics to compute. Options: 'accuracy', 'f1'.
                           Default is ["accuracy", "f1"].
    - f1_average (str): Averaging method for F1-score. See sklearn.metrics.f1_score for options.
                        Default is "macro".
    - f1_zero_division (int or float): Value to return when there is a zero division.
                                       Can be 0 or 1. Default is 1, meaning an undefined
                                       F-score (due to no true/predicted samples) will be 1.0.

    Returns:
    - result (dict): Computed metric value or a dictionary containing different evaluation metrics.
    """
    metrics = {}

    if not isinstance(inputs, np.ndarray) or not isinstance(
        outputs, np.ndarray
    ):
        raise TypeError("Inputs/outputs must be torch.Tensor or numpy.ndarray for predictions.")

    for metric in metrics_list:

        if metric not in ["accuracy", "f1"]:
            raise ValueError(
                f"Invalid metric: '{metric}'. Choose from 'accuracy' or 'f1'."
            )

        if metric == "f1":
            # For binary classification, you might want to specify pos_label if not 0 or 1
            # and average='binary' or 'weighted'. 'macro' treats both classes equally.
            f1 = f1_score(inputs, outputs, average=f1_average, zero_division=f1_zero_division, labels=labels)
            metrics["f1"] = f1

        if metric == "accuracy":
            accuracy = accuracy_score(inputs, outputs)
            metrics["accuracy"] = accuracy

    return metrics


# def calc_metric_similarity(targets, outputs, metrics_list=['SSIM'], jaccard_average:str='weighted',
#                            jaccard_zero_division:int=1,
#                            ):
#     """
#     Evaluate the performance of a recontructive model using various metrics.

#     Args:
#     - targets (np.ndarray): Original input images OR target series predictions.
#     - outputs (np.ndarray): Reconstructed output images OR series predictions.
#     - metric (str): Metric to compute. Options: 'MSE', 'BCE', 'MAE', 'SSIM', 'PSNR'.

#     Returns:
#     - metrics (dict): Computed metric value or a dictionary containing different evaluation metrics.

#     Example:
#     >>> arr1, arr2 = np.random.rand(1, 3, 16, 16), np.random.rand(1, 3, 16, 16)
#     >>> metrics = calc_metric_reconstruction(arr1, arr2, metrics_list=['SSIM'])

#     """
#     metrics = {}
#     metrics_list_ref = ['MSE', 'RMSE', 'MAE', 'Jaccard', 'SSIM', 'DICE', 'PSNR', 'MAPE', 'SMAPE', 'R-squared']

#     for metric in metrics_list:

#         # check metrics
#         if metric not in metrics_list_ref:
#             raise ValueError(f"Invalid metric. Choose among {metrics_list_ref}")

#         # Mean Squared Error (MSE)
#         if metric=='MSE':
#             metrics['MSE'] = mean_squared_error(targets, outputs)

#         # Root MSE (RMSE)
#         if metric == 'RMSE':
#             mse = mean_squared_error(targets, outputs)
#             metrics['RMSE'] = np.sqrt(mse)

#         # Mean Absolute Error (MAE)
#         if metric=='MAE':
#             metrics['MAE'] = mean_absolute_error(targets, outputs)

#         if metric=='Jaccard':
#             if len(targets.shape)>1:
#                 targets = targets.ravel()
#             if len(outputs.shape)>1:
#                 outputs = outputs.ravel()
#             metrics['Jaccard'] = jaccard_score(targets, outputs, average=jaccard_average, zero_division=jaccard_zero_division)

#         # Structural Similarity Index (SSIM)
#         if metric=='SSIM':
#             multichannel = False if len(targets.shape) == 2 else True
#             ssim_value = ssim(targets, outputs, channel_axis=multichannel)
#             metrics['SSIM'] = np.mean(ssim_value)

#         # DICE Scores
#         if metric=='DICE':
#             intersection = np.logical_and(targets, outputs)
#             union = np.logical_or(targets, outputs)
#             metrics['DICE'] = (2.0 * intersection.sum()) / (union.sum() + 1e-8)

#         # Peak Signal-to-Noise Ratio (PSNR)
#         if metric=='PSNR':
#             metrics['PSNR'] = psnr(targets, outputs)

#         # absolute percentage error (MAPE)
#         if metric=='MAPE':
#             absolute_percentage_errors = abs((targets - outputs) / targets)
#             metrics['MAPE'] = np.mean(absolute_percentage_errors) * 100

#         # symmetric absolute percentage error (SMAPE)
#         if metric=='SMAPE':
#             symmetric_absolute_percentage_errors = 2 * np.abs(targets - outputs) / (np.abs(targets) + np.abs(outputs))
#             metrics['SMAPE'] = np.mean(symmetric_absolute_percentage_errors) * 100

#         # R-squared
#         if metric=='R-squared':
#             metrics['R-squared'] = r2_score(targets, outputs)

#     return metrics

import numpy as np
from skimage.metrics import (
    structural_similarity as ssim,
    peak_signal_noise_ratio as psnr,
)
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    jaccard_score,
    r2_score,
)
from scipy.stats import entropy
from sklearn.preprocessing import normalize
from scipy.spatial.distance import cdist
import cv2


def mutual_information(hgram):
    """Compute mutual information from a 2D histogram."""
    pxy = hgram / float(np.sum(hgram))
    px = np.sum(pxy, axis=1)  # marginal for x
    py = np.sum(pxy, axis=0)  # marginal for y
    px_py = np.outer(px, py)
    nonzero = pxy > 0
    return np.sum(pxy[nonzero] * np.log(pxy[nonzero] / px_py[nonzero]))


def normalized_mutual_information(img1, img2, bins=256):
    """Calculate Normalized Mutual Information (NMI) between two images."""
    hgram, _, _ = np.histogram2d(img1.ravel(), img2.ravel(), bins=bins)
    H_img1 = entropy(np.sum(hgram, axis=1))
    H_img2 = entropy(np.sum(hgram, axis=0))
    MI = mutual_information(hgram)
    return 2.0 * MI / (H_img1 + H_img2)


def calc_FRE(landmarks_fixed, landmarks_moving_transformed):
    """Calculate Fiducial Registration Error (FRE) between fixed and transformed landmarks."""
    return np.sqrt(
        np.mean(
            np.sum(
                (landmarks_fixed - landmarks_moving_transformed) ** 2, axis=1
            )
        )
    )


def calc_metric_similarity(
    targets,
    outputs,
    metrics_list=["SSIM"],
    jaccard_average="weighted",
    jaccard_zero_division=1,
    landmarks_fixed=None,
    landmarks_moving_transformed=None,
):
    """
    Evaluate the performance of a reconstruction or registration using various metrics.

    Args:
    - targets (np.ndarray): Target image or data (e.g., fixed image).
    - outputs (np.ndarray): Output image or data (e.g., moving image or predicted).
    - metrics_list (list[str]): Metrics to compute. Options:
        ['MSE' (lower better), 'RMSE' (lower better), 'MAE' (lower better),
         'SSIM' (higher better), 'PSNR' (higher better), 'MAPE' (lower better),
         'SMAPE' (lower better), 'R-squared' (higher better),
         'Jaccard' (higher better), 'DICE' (higher better),
         'NMI' (higher better), 'FRE' (lower better)]
    - jaccard_average (str): Jaccard average method.
    - jaccard_zero_division (int): Handling of zero-division in Jaccard.
    - landmarks_fixed (np.ndarray): Nx2 or Nx3 array of fixed landmarks.
    - landmarks_moving_transformed (np.ndarray): Nx2 or Nx3 array of transformed landmarks.

    Returns:
    - metrics (dict): Dictionary of computed metric values.
    """
    metrics = {}
    metrics_list_ref = [
        "MSE",
        "RMSE",
        "MAE",
        "Jaccard",
        "SSIM",
        "DICE",
        "PSNR",
        "MAPE",
        "SMAPE",
        "R-squared",
        "NMI",
        "FRE",
    ]

    for metric in metrics_list:

        if metric not in metrics_list_ref:
            raise ValueError(f"Invalid metric. Choose among {metrics_list_ref}")

        if metric == "MSE":
            metrics["MSE"] = mean_squared_error(targets, outputs)

        if metric == "RMSE":
            mse = mean_squared_error(targets, outputs)
            metrics["RMSE"] = np.sqrt(mse)

        if metric == "MAE":
            metrics["MAE"] = mean_absolute_error(targets, outputs)

        if metric == "Jaccard":
            metrics["Jaccard"] = jaccard_score(
                targets.ravel(),
                outputs.ravel(),
                average=jaccard_average,
                zero_division=jaccard_zero_division,
            )

        if metric == "SSIM":
            multichannel = len(targets.shape) > 2
            ssim_value = ssim(
                targets, outputs, channel_axis=-1 if multichannel else None
            )
            metrics["SSIM"] = np.mean(ssim_value)

        if metric == "DICE":
            intersection = np.logical_and(targets, outputs)
            union = np.logical_or(targets, outputs)
            metrics["DICE"] = (2.0 * intersection.sum()) / (union.sum() + 1e-8)

        if metric == "PSNR":
            metrics["PSNR"] = psnr(targets, outputs)

        if metric == "MAPE":
            absolute_percentage_errors = np.abs(
                (targets - outputs) / np.clip(targets, 1e-8, None)
            )
            metrics["MAPE"] = np.mean(absolute_percentage_errors) * 100

        if metric == "SMAPE":
            symmetric_absolute_percentage_errors = (
                2
                * np.abs(targets - outputs)
                / (np.abs(targets) + np.abs(outputs) + 1e-8)
            )
            metrics["SMAPE"] = (
                np.mean(symmetric_absolute_percentage_errors) * 100
            )

        if metric == "R-squared":
            metrics["R-squared"] = r2_score(targets, outputs)

        if metric == "NMI":
            metrics["NMI"] = normalized_mutual_information(targets, outputs)

        if metric == "FRE":
            if landmarks_fixed is None or landmarks_moving_transformed is None:
                raise ValueError(
                    "FRE requires both landmarks_fixed and landmarks_moving_transformed."
                )
            metrics["FRE"] = calc_FRE(
                landmarks_fixed, landmarks_moving_transformed
            )

    return metrics

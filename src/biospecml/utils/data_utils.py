import pandas as pd
import numpy as np

def check_nans_infs(X, check_nans: bool = True, check_infs: bool = True, check_zeros: bool = False,
                    check_negatives: bool = False, replace_zeros: bool = False, replace_nans: bool = True,
                    replace_inf: bool = True, replace_negatives: bool = False,
                    replace_value: float = 1.4013e-45):
    """
    Check for NaNs, zeros, infinities, and negative values in the input array or DataFrame, and optionally replace them.

    Args:
        X (pd.DataFrame | np.ndarray): Input DataFrame or array.
        check_nans (bool): Check for NaNs. Default is True.
        check_infs (bool): Check for infinities. Default is True.
        check_zeros (bool): Check for zeros. Default is False.
        check_negatives (bool): Check for negative values. Default is False.
        replace_zeros (bool): Replace zeros with `replace_value`. Default is True.
        replace_nans (bool): Replace NaNs with `replace_value`. Default is True.
        replace_inf (bool): Replace infinities with `replace_value`. Default is True.
        replace_negatives (bool): Replace negative values with `replace_value`. Default is True.
        replace_value (float): Value used for replacement of zeros, NaNs, infinities, and negative values.

    Returns:
        np.ndarray | pd.DataFrame: Processed array or DataFrame.
    """

    replaced_status = False

    # Convert input to DataFrame if it's a NumPy array
    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X)

    # Check if NaNs, zeros, infinities, or negatives are present
    if check_nans and X.isna().any().any():
        print("check_nans_infs(): NaNs detected in data.")
        if replace_nans:
            X = X.fillna(replace_value)
            replaced_status = True
            print("check_nans_infs(): Replaced NaNs in data.")

    if check_zeros and (X == 0).any().any():
        print("check_nans_infs(): Zeros detected in data.")
        if replace_zeros:
            X = X.replace(0, replace_value)
            replaced_status = True
            print("check_nans_infs(): Replaced zero in data.")

    if check_infs and np.isinf(X.values).any():
        print("check_nans_infs(): Infinities detected in data.")
        if replace_inf:
            X = X.replace([np.inf, -np.inf], replace_value)
            replaced_status = True
            print("check_nans_infs(): Replaced infs and/or -infs in data.")

    if check_negatives and (X < 0).any().any():
        print("check_nans_infs(): Negative values detected in data.")
        if replace_negatives:
            X = X.mask(X < 0, replace_value)
            replaced_status = True
            print("check_nans_infs(): Replaced negative values in data.")

    # Return either DataFrame or array based on the input type
    return X if isinstance(X, np.ndarray) else X.values, replaced_status


def calc_z_score(arr):
    """
    A simple functions to calculate Z-scores.
    
    """
    mean_arr = np.mean(arr)
    std_arr = np.std(arr)
    z_scores = (arr - mean_arr) / std_arr
    return z_scores
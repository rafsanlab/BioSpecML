import numpy as np
import pandas as pd

def norm_peak(X, peak_range:range=(1680, 1620), mode:str='intensity', 
              epsilon:float=1.4013e-45, wn=None):
    """
    Applies normalisation on a dataframe based on wavenumber.
    mode 'intensity' use the mean of the peak_range while
    mode 'range' use the whole slice of peak_range.
    if X is a dataframe; rows is the wn, and columns is the samples,
    if X is a np.ndarray; use sp.
    epsilon is a constant added to prevent division by zero.

    """
    if isinstance(X, pd.DataFrame):
        start = peak_range[0]
        end = peak_range[1]
        if mode == 'intensity':
            norm_values = X.loc[start:end].mean().mean()
        elif mode == 'range':
            norm_values = X.loc[start:end].mean()

        X = X / (norm_values + epsilon)

    elif isinstance(X, np.ndarray):
        if wn is None:
            raise Exception('Provide *wn if X is np.ndarray.')
        start = peak_range[0]
        end = peak_range[1]
        indices = np.where((wn >= start) & (wn <= end))[0]
        
        if mode == 'intensity':
            norm_values = np.mean(X[indices])
        elif mode == 'range':
            norm_values = np.mean(X[indices], axis=0)

        X = X / (norm_values + epsilon)

    return X
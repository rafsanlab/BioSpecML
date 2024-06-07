import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize

def norm_peak(X, peak_range:tuple=(1620, 1680), mode:str='intensity', 
              epsilon:float=1.4013e-45, wn=None, verbose:bool=False):
    """
    Applies normalisation on a dataframe or np array based on wavenumber range.
    
    Args:
    - X : matrix of spectral data;
        - if X is a dataframe;
            - rows is the wn, and columns is the samples,
            - set wn as index
            - peak_range must according to wn order
        - if X is a np.ndarray, use sp and provide wn
    - mode : mode of normalisation;
        - mode 'intensity' use the mean of the peak_range
        - mode 'range' use the whole slice of peak_range.
    - epsilon : constant added to prevent division by zero.

    Returns:
    - X

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

    if verbose:
        print('Wavenumber slice:', (wn[indices]))
        print('Mean:', norm_values)
    
    return X


def norm_spectra(sp, method='vector'):
    """
    Provide sp (wn, h*w).

    """
    if method=='vector':
        sp_norm = normalize(sp, axis=0, norm='l2')
    return sp_norm

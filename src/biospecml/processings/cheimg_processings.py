import numpy as np
import pandas as pd

def calc_snr(dfX, signal_range, noise_range):
    """
    Calculate signal to noise (SNR) ratio based on mean signal / std noise.

    Args:
        dfX(pd.DataFrame): matrix of a df.
        signal_range(tuple): wavenumber range from df columns.
        noise_range(tuple): wavenumer ranger from df columns.
    Return:
        SNR ratio(np.array).
    """

    numeric_cols = pd.to_numeric(dfX.columns) # convert columns to numbers
    signal = dfX.loc[:, (numeric_cols >= signal_range[1]) & (numeric_cols <= signal_range[0])]
    signal = signal.mean(axis=1) # calculate signal
    noise = dfX.loc[:, (numeric_cols >= noise_range[1]) & (numeric_cols <= noise_range[0])]
    noise = noise.std(axis=1) # calculate noise
    snr = np.where(noise==0, 0, signal/noise) # calculate SNR ratio
    return snr


def calc_outliers_threshold(snr_column, n):
    """
    Calculate the outliers threshold of all spectra dataframe based on the formula
    = mean + n * SD, use this value to filter spectra outliers from the dataset.

    Args:
        snr_column(pd.core.series.Series): SNR column.
        n(int): SD multiplier.
    Return:
        outliers_threshold(float): the threshold value.
    """

    outliers_threshold = snr_column.mean() + (n * snr_column.std()) # mean+3xSD
    return outliers_threshold
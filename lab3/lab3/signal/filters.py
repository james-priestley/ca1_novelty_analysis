"""Filter functions for signals. Functions in this module should take as their
first argument an array-like object of size (n_ROIs, n_timepoints), and should
return results in an identical format. If the input is a DataFrame, this
should be preserved (along with its index)."""


import pandas as pd


def maxmin_filter(signal, window=300, sigma=5, min_periods=0.2):
    """Calculate baseline as the rolling maximum of the rolling minimum of the
    smoothed trace

    Parameters
    ----------
    signal : array, size (n_ROIs, n_timepoints)
    window : int
        Optional, size of the rolling window for max/min/smoothing
    sigma : int
        Standard deviation of the gaussian smoothing kernel
    min_periods : float [0, 1]
        Optional, percentage of values in each window that must be non-NaN to
        return a non-NaN value in the baseline
    """

    kwargs = {'window': window, 'min_periods': int(window * min_periods),
              'center': True, 'axis': 1}

    smooth_signal = pd.DataFrame(signal).rolling(
        win_type='gaussian', **kwargs).mean(std=sigma)

    return smooth_signal.rolling(**kwargs).min().rolling(**kwargs).max()


def smooth_quantile_filter(signal, quantile=0.08, window=300, sigma=5,
                           min_periods=0.2):

    kwargs = {'window': window, 'min_periods': int(window * min_periods),
              'center': True, 'axis': 1}

    smooth_signal = pd.DataFrame(signal).rolling(
        win_type='gaussian', **kwargs).mean(std='sigma')

    return smooth_signal.rolling(**kwargs).quantile(quantile)

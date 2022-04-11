import numpy as np 
import scipy as sp
import scipy.ndimage

import scipy.stats as stats

def nan_zscore(array):
    """Compute the zscore of an array which may contain nans.
    Parameters
    ----------
    array : array-like
        Array to compute zscores on
    """
    return (array - np.nanmean(array))/np.nanstd(array)

def nonparametric_nan_zscore(array):
    """Compute the nonparametric zscore of an array which may contain nans.
    Parameters
    ----------
    array : array-like
        Array to compute nonparametric zscores on. Use median and IQR 
        instead of mean and std. Default to False.
    """
    return (array - np.nanmedian(array))/stats.iqr(array, nan_policy='omit')    

def nan_filter(array, filter_func, **filter_kwargs):
    """Implements a normalized convolution. This is only well-defined
    for LINEAR (i.e., convolution-based) filters filter_func.
    
    Example
    ----------
    >>> U=sp.randn(10,10)        
    >>> U[U>2]=np.nan              
    >>> U_filtered = nan_filter(U, sp.ndimage.gaussian_filter, 
                            sigma=2, truncate=4)
    """

    V = np.nan_to_num(array.copy())
    V_filtered = filter_func(V, **filter_kwargs)

    W = np.ones_like(array)
    W[np.isnan(array)] = 0
    W_filtered = filter_func(W, **filter_kwargs)

    return V_filtered/W_filtered




import numpy as np
from scipy.signal import resample


def calc_cutoff(length, fs, max_trunc=None):
    """DSP trick to speed up resampling

    Parameters
    ----------
    length : int
        Length of original signal
    fs : float
        Sampling rate of original signal
    max_trunc : float, optional
        Maximum allowable truncation (1 / [fs units])
    """
    if max_trunc is None:
        return None
    else:
        max_trunc_samples = fs * max_trunc
        exponent = int(np.log2(max_trunc_samples))
        assert exponent > 0
        cutoff = int(length / (2 ** exponent)) * 2 ** exponent
        return cutoff


def fast_resample(x, num, t=None, axis=0, fs=None, max_truncation=None,
                  truncate_output=False, **kwargs):
    """Uses a DSP trick to speed up resampling

    Parameters
    ----------
    x : array
        Signal to resample
    num : int
        Number of samples in output
    t : array, optional
        Times to sample
    axis : int, optional
        Defaults to 0
    fs : float, optional
        Sampling rate of original signal
    max_truncation : float, optional
        Maximum allowable truncation (1 / [fs units])
    truncate_output : bool, optional
        Apply same trick to the output (NOTE: this results in at most 2x
        max_truncation)

    Notes
    -----
    This only supports downsampling for now. Either t or fs must be specified!
    Sampling rate should be about uniform
    """

    if (t is None) and (fs is None):
        raise ValueError("One of 't' or 'fs' must be specified!")

    length = x.shape[axis]
    if num > length:
        raise NotImplementedError("Only downsampling is supported for now")

    if t is not None:
        fs = 1 / np.mean(np.diff(t))

    cutoff = calc_cutoff(length, fs, max_trunc=max_truncation)

    truncated_x = np.moveaxis(np.moveaxis(x, axis, 0)[:cutoff], 0, axis)

    if truncate_output:
        resample_factor = num / length
        resampled_fs = fs * resample_factor
        output_length = calc_cutoff(num, resampled_fs,
                                    max_trunc=max_truncation)
        return resample(truncated_x, output_length, t=t, axis=axis, **kwargs)
    else:
        return resample(truncated_x, num, axis=axis, t=t, **kwargs)

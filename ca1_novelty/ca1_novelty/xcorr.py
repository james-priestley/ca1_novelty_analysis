"""Module containing functions for spatial crosscorrelation analysis"""

from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy.stats import pearsonr


def analyze_spatial_drift(expt, max_shift=25, min_corr=0.25, min_frac=0.33,
                          include_all_trials=False, **kwargs):
    """
    Main function for spatial drift analysis.

    Parameters
    ----------
    max_shift : int
    min_corr : float [0, 1]
    min_frac : float [0, 1]
    include_all_trials : bool
    """

    rasters = expt.block_rasters()

    # iterate over trial blocks
    results = []
    for (block, context), block_raster in rasters.groupby(
            ['block_num', 'context']):

        if block > 3:
            continue

        rstack = np.stack([l.values for _, l in block_raster.groupby('lap')])
        xcorr_tensor = pairwise_xcorr(rstack, max_shift=max_shift, **kwargs)

        if not include_all_trials:
            valid_trials = np.mean(
                np.max(xcorr_tensor, axis=-1) > min_corr, axis=0) > min_frac
            xcorr_tensor[~valid_trials] = np.nan
            xcorr_tensor[:, ~valid_trials] = np.nan

        displacements = center_of_mass(xcorr_tensor) - max_shift
        drift = calc_drift_score(displacements)

        results.append({
            "expt": expt.trial_id,
            "day": expt.day,
            "context": context,
            "block": block,
            "displacements": displacements,
            "xcorr_tensor": xcorr_tensor,
            "drift": drift,
            "set_id": expt.set_id,
        })

    return pd.DataFrame(results)


def calc_drift_score(d):
    """Calculate drift score. <explanation>

    Parameters
    ----------
    d : array (n_trials, n_trials)
        Array of pairwise trial displacements. d[i, j] gives
        the shift applied to trial i that maximizes the
        correlation with trial j.

    Returns
    -------
    score : float
    """

    # symmetrize displacements (using upper triangle)
    sym_d = np.triu(d)
    idx = np.tril_indices(sym_d.shape[0])
    sym_d[idx] = sym_d.T[idx]

    return np.nanmedian(sym_d, axis=0)


def _population_xcorr(args):
    "Helper function for multi-processing"
    return population_xcorr(args[0], args[1], **args[2])


def pairwise_xcorr(trial_tensor, **kwargs):
    """
    Parameters
    ----------

    Returns
    -------
    """

    def _iter_trial_pairs(rasters, kwargs):
        for t1, r1 in enumerate(rasters):
            for t2, r2 in enumerate(rasters[(t1 + 1):]):
                yield (r1, r2, kwargs)

    p = Pool(8)
    xcorrs = p.map(_population_xcorr, _iter_trial_pairs(trial_tensor, kwargs))
    p.close()
    p.join()

    xcorr_tensor = np.zeros((trial_tensor.shape[0], trial_tensor.shape[0],
                             xcorrs[0].shape[0]))
    xcorr_tensor[:] = np.nan
    for cidx, (t1, t2) in enumerate(
            zip(*np.triu_indices(trial_tensor.shape[0], 1))):
        xcorr_tensor[t1, t2] = xcorrs[cidx]
        xcorr_tensor[t2, t1] = xcorrs[cidx][::-1]

    return xcorr_tensor


def population_xcorr(a, b, max_shift=25, crop=True):
    """Given two population of neural responses a and b, compute the population
    vector cross-correlation over a range of lags, by rotating each row of a
    and correlating the flattened a and b arrays.

    Parameters
    ----------
    a : array, shape (n_rois, n_bins)
    b : array, shape (n_rois, n_bins)
    max_shift : int, optional
        Range of shifts to compute cross-correlation. Defaults to 25
    crop : bool
        Whether to trim off the area of the curves that has been circularly
        appended before computing the correlation

    Returns
    -------
    xcorr : array, shape (max_shift * 2 + 1,)
        Spatial cross-correlation between the tuning curves and reference.
    """

    # TODO - we should rotate the reference!

    deltas = np.arange(-1 * max_shift, max_shift + 1)[::-1]
    xcorr = []
    for d in deltas:
        a_shifted = np.roll(a, d, axis=1)
        if crop:
            if d < 0:
                xcorr.append(pearsonr(a_shifted[:, :d].flatten(),
                                      b[:, :d].flatten())[0])
                continue
            elif d > 0:
                xcorr.append(pearsonr(a_shifted[:, d:].flatten(),
                                      b[:, d:].flatten())[0])
                continue

        xcorr.append(pearsonr(a_shifted.flatten(),
                              b.flatten())[0])

    return np.asarray(xcorr)


def compute_block_xcorr(rasters, loo=True, num_template_trials=15, **kwargs):
    """
    Parameters
    ----------
    rasters :
    loo : bool, optional
        Whether to recompute the reference when compute cross-correlation for
        trials that are in the reference block, i.e. "leave-one-out". Defaults
        to True.
    num_template_trials : int, optional
        Number of trials to include in the reference. Reference trials are
        counted from the end of the trial block. Defaults to 15.

    Additional keyword arguments are passed directly to `population_xcorr`.
    """
    trial_idx = np.arange(rasters.shape[0])

    if num_template_trials is None:
        is_template = np.array([True] * len(trial_idx))
    else:
        is_template = np.in1d(trial_idx, trial_idx[-num_template_trials:])

    template = rasters[is_template].mean(axis=0)

    xcorr = []
    for trial_num, trial_curves in enumerate(rasters):

        if is_template[trial_num] & loo:
            template = rasters[(trial_idx != trial_num)
                               & is_template].mean(axis=0)

        xcorr.append(population_xcorr(trial_curves, template), **kwargs)

    return np.stack(xcorr)  # n_trials x n_lags


def center_of_mass(a, axis=-1):
    """Compute the center of mass of the curve."""
    if len(a.shape) > 1:
        return np.apply_along_axis(center_of_mass, axis, a)
    return (a * np.arange(len(a))).sum() / a.sum()

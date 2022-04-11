import numpy as np
import pandas as pd
import scipy.stats as sp

from lab3.signal.spikes import OasisAR1Spikes
from platz.tuning import SimpleSpatialTuning

from ca1_novelty.xcorr import center_of_mass as _center_of_mass


def btsp_kernel(t):
    return np.exp(-t / 0.69) * np.heaviside(t, 0) \
           + np.exp(t / 1.31) * np.heaviside(-t, 1)


def position_btsp(plateau_x, input_xs, velocity):
    """For a plateau that occurs at plateau_x when the animal is moving
    at the specified velocity, calculate the relative potentiation of
    inputs at locations input_xs"""

    dt = (input_xs - plateau_x) / velocity
    return btsp_kernel(dt)


def prep_tuning_inputs(values, pos, laps, max_pos, n_bins=100):

    valid_samples = ~np.isnan(pos)
    pos = (np.floor((pos / max_pos) * n_bins)).astype('int')

    return (pd.DataFrame(values[valid_samples]).T,
            pos[valid_samples],
            laps[valid_samples])


def reconvolve_spikes(spikes, tau_d=0.4, tau_r=0.06, dt=1e-3):

    tau_d = tau_d / dt
    tau_r = tau_r / dt

    g = [np.exp(-1 / tau_d) + np.exp(-1 / tau_r),
         -1 * np.exp(-1 / tau_d) * np.exp(-1 / tau_r)]

    c = np.zeros(len(spikes))
    c = spikes.astype(float)
    for i in range(2, len(spikes)):
        c[i] += g[0] * c[i - 1] + g[1] * c[i - 2]

    return sp.zscore(c)


def subsample_calcium(c, snr, fs=10, dt=1e-3):

    skip = int(1 / (fs * dt))
    baseline = np.min(c)

    subsampled_c = c[::skip]
    return subsampled_c + np.random.normal(
        -baseline, 1 / snr, size=subsampled_c.shape)


def infer_spikes(calcium, fs, **kws):
    return OasisAR1Spikes(fs=fs, **kws).calculate(
        pd.DataFrame(calcium).T).values.squeeze()


def poisson_spikes(firing_rate, dt):
    return np.random.uniform(0, 1, size=firing_rate.shape) < (firing_rate * dt)


def compute_raster(signal, pos, laps, track_length, sigma=3, n_bins=100):
    tuning_inputs = prep_tuning_inputs(signal, pos, laps, track_length,
                                       n_bins=n_bins)
    return SimpleSpatialTuning(sigma=3).calculate_rasters(
        *tuning_inputs, max_position=n_bins).values.astype('float')


def center_of_mass(a, threshold=0.1):
    a[a < threshold * np.max(a)] = 0
    return _center_of_mass(a)


def calc_first_lap_shift(r):

    if (np.sum(r[0]) == 0) | (np.sum(r[1:]) == 0):
        return np.nan

    first_lap = center_of_mass(r[0])
    remaining = center_of_mass(r[1:].mean(axis=0))

    return first_lap - remaining


def calc_field_width(tuning_curve, thres=0.05):

    max_rate = np.max(tuning_curve)
    max_loc = np.argmax(tuning_curve)

    supthres = tuning_curve > thres * max_rate
    try:
        end = np.where(supthres[max_loc:] == 0)[0][0] + max_loc
    except Exception:
        end = len(tuning_curve)
    try:
        start = np.where(supthres[:max_loc] == 0)[0][-1]
    except Exception:
        start = 0

    return end - start

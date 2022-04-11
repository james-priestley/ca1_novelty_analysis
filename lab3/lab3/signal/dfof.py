"""Methods for calculating dF/F. Implement custom strategies by subclassing
DFOFStrategy and filling in the details."""

from abc import abstractmethod
# from .base import BaseSignalTransformer

import numpy as np
import pandas as pd
# import multiprocessing as mp

from lab3.signal import filters
from lab3.core import Automorphism


class DFOFStrategy(Automorphism):

    """Base class for dF/F calculation strategies. Derived strategies should
    implement the _filter_final_signal and _calculate_baseline methods."""

    name = "DFOF"

    def apply_to(self, experiment, channel='Ch2', label=None, **kwargs):
        """Apply dF/F analysis to an experiment object.

        Parameters
        ----------
        experiment : lab3.experiment.base.ImagingExperiment
            Instance of ImagingExperiment to analyze
        channel :
        label :

        Additional keyword arguments are passed directly to `calculate`.
        """
        sig_dict = self._load_signals(experiment, channel=channel, label=label)
        return self.calculate(**sig_dict, **kwargs)

    def calculate(self, signal, *args, **kwargs):
        """Calculate dF/F

        Parameters
        ----------
        signal : array-like (n_rois, n_samples)
            2D array of roi signals

        Returns
        -------
        dfof : pd.DataFrame (n_rois, n_samples)
            2D array of roi dF/F
        """

        if len(signal.shape) != 2:
            raise ValueError("Input signal must be 2D")

        try:
            del self.baseline
        except AttributeError:
            pass

        base = self.calculate_baseline(signal, *args, **kwargs)
        return self._filter_final_signal(
            (signal - base) / base, *args, **kwargs)

    # ----------------- Implement these methods to subclass ----------------- #

    @abstractmethod
    def _filter_final_signal(self, signal):
        """Apply smoothing to the final trace."""

    @abstractmethod
    def _calculate_baseline(self, signal):
        """Estimate baseline for dFOF calculation."""

    def calculate_baseline(self, signal, *args, **kwargs):
        """
        Parameters
        ----------
        signal : array-like (n_rois, n_samples)
            2D array of roi signals

        Returns
        -------
        baseline : pd.DataFrame (n_rois, n_samples)
            2D array of roi baselines
        """
        try:
            return self.baseline
        except AttributeError:
            self.baseline = self._calculate_baseline(signal, *args, **kwargs)
            return self.baseline

    # ------------------------------- Optional ------------------------------ #

    def _load_signals(self, experiment, channel, label):
        """By default only the raw signal dataframe is passed to `calculate`
        when using `apply`. If you override `calculate` to take additional
        arguments in your custom strategy, you should load and add them to the
        dictionary here, which will be unpacked as arguments to `calculate`.
        """
        sig_dict = {
            'signal': experiment.signals(signal_type='raw', channel=channel,
                                         label=label, max_frame=None)
        }
        return sig_dict


class SlowTrendMixin:

    """Mixin class that provides a final detrending step at the end of
    dF/F calculation, calculated as a rolling median filter."""

    # TODO should this override the calculate method? It would make composition
    # easier

    def calculate_slow_trend(self, signal, window=None, min_periods=None):
        """
        Parameters
        ----------
        signal : array-like (n_rois, n_samples)
            2D array of roi signals
        window : int, optional
            Size of window in frames to compute rolling median
        min_periods : float (0, 1], optional
            Fraction of frames in rolling window that must be non-NaN to return
            a non-NaN value in the filtered signal

        Returns
        -------
        slow_trend : pd.DataFrame (n_rois, n_samples)
            2D array of roi slow trends (rolling median)
        """
        try:
            return self.slow_trend
        except AttributeError:
            if window is None:
                return pd.DataFrame(np.zeros(signal.shape))
            else:
                self.slow_trend = pd.DataFrame(signal).rolling(
                    window=window, min_periods=int(window * min_periods),
                    center=True, axis=1).median()
                return self.slow_trend


class NaNFillMixin:

    """Mixin class to add NaN-filling functionality to dF/F calculation."""

    # TODO this should override the calculate method and fill in NaNs
    # How to handle extra parameters?

    pass


class QuantileDFOF(DFOFStrategy):

    """Basic dF/F, with baseline calculated as a rolling percentile of the
    smoothed raw trace. No filtering is applied to the trace in the final
    calculation.

    Parameters
    ----------
    quantile : float (0, 1], optional
        Quantile for baseline calculation. Defaults to 0.08
    window : int, optional
    sigma : float, optional
        Standard deviation (in frames) of gaussian smoothing kernel, applied
        prior to the quantile filter in the baseline calculation.
        Defaults to 5.
    min_periods : float (0, 1], optional

    Attributes
    ----------
    baseline : pd.DataFrame
    """

    def __init__(self, quantile=0.08, window=300, sigma=5, min_periods=0.2):
        self.quantile = quantile
        self.window = window
        self.sigma = sigma
        self.min_periods = min_periods

    def _filter_final_signal(self, signal):
        """No final filtering"""
        return signal

    def _calculate_baseline(self, signal):
        """Smooth signal with a Gaussian, then apply rolling quantile filter"""

        kwargs = {'window': self.window,
                  'min_periods': int(self.window * self.min_periods),
                  'center': True,
                  'axis': 1}

        smooth_df = pd.DataFrame(signal).rolling(
            win_type='gaussian', **kwargs).mean(std=self.sigma)

        return pd.DataFrame(smooth_df).rolling(**kwargs).\
            quantile(self.quantile)


class CustomFilterDFOF(DFOFStrategy):

    """Pass functions for baseline and filter methods and their respective
    keyword arguments.

    Parameters
    ----------
    base_func, filt_func : function
        A function that takes a 2D array of signals and returns a 2D array of
        baselines (base_func), final filtered signals (filt_func)
    base_kws, filt_kws : dict, optional
        Additional keyword arguments for base_func and filt_func, stored as
        key, value pairs in a dictionary. Optional.

    Attributes
    ----------
    baseline : pd.DataFrame
    """

    def __init__(self, base_func, filt_func, base_kws={'axis': 1},
                 filt_kws={'axis': 1}):
        self._filter_final_signal = lambda x: filt_func(x, **filt_kws)
        self._calculate_baseline = lambda x: base_func(x, **base_kws)


class JiaDFOF(DFOFStrategy, SlowTrendMixin):

    """Implements the dF/F calculation method of Jia et al 2010 (Nat. Protocols)
    with additional options to detrend slow changes.

    Parameters
    ----------
    t1 : int, optional
        Window size in frames for initial smoothing. A good value is ~3 sec
        of imaging time. Defaults to 90 frames.
    t2 : int, optional
        Window size in frames for rolling minimum baseline. A good value is ~60
        sec of imaging time. Defaults to 1800 frames.
    exp : float, optional
        Time constamt for exponential filtering of final trace.
        Not implemented
    slow_trend_window : int, optional
        If passed, the final trace is median-filtered in a rolling window of
        this size to remove slow changes (e.g. due to photobleaching). This
        is conventionally larger than the baseline window. Defaults to None
        (no detrending is applied).
    min_periods_t1, min_periods_t2, min_periods_slow : float (0, 1], optional
        Fraction of frames in rolling t1, t2, and slow_trend windows that must
        be non-NaN to return a non-NaN value.

    Attributes
    ----------
    baseline : pd.DataFrame
    slow_trend : pd.DataFrame
    """

    def __init__(self, t1=90, t2=1800, exp=None, min_periods_t1=0.2,
                 min_periods_t2=0.2, slow_trend_window=None,
                 min_periods_slow=0.2):
        self.t1 = t1
        self.t2 = t2
        self.exp = exp
        self.min_periods_t1 = min_periods_t1
        self.min_periods_t2 = min_periods_t2
        self.slow_trend_window = slow_trend_window
        self.min_periods_slow = min_periods_slow

    def _filter_final_signal(self, signal):
        """Filter with exponential kernel."""

        # apply exponential filter
        if self.exp is None:
            pass
        else:
            raise NotImplementedError

        # remove slow changes
        try:
            del self.slow_trend
        except AttributeError:
            pass

        return signal - self.calculate_slow_trend(
            signal, self.slow_trend_window, self.min_periods_slow)

    def _calculate_baseline(self, signal):
        # first smooth with t1 rolling window
        kwargs = {'center': True, 'axis': 1}
        smooth_signal = pd.DataFrame(signal).rolling(
            window=self.t1, min_periods=int(self.min_periods_t1 * self.t1),
            **kwargs).mean()

        # then minimize with t2 rolling window
        return smooth_signal.rolling(
            window=self.t2, min_periods=int(self.min_periods_t2 * self.t2),
            **kwargs).min()


class MaxminDFOF(DFOFStrategy):
    """
    Parameters
    ----------
    window : int, optional
    sigma : float, optional
    min_periods : float (0, 1], optional

    Attributes
    ----------
    baseline : pd.DataFrame
    """

    def __init__(self, window=1800, sigma=90, min_periods=0.2):
        self.window = window
        self.sigma = sigma
        self.min_periods = min_periods

    def _filter_final_signal(self, signal):
        return signal

    def _calculate_baseline(self, signal):
        """Apply maxmin filter to signal to find baseline"""
        return filters.maxmin_filter(signal, window=self.window,
                                     sigma=self.sigma,
                                     min_periods=self.min_periods)


class Suite2pDFOF(DFOFStrategy):

    """Class for calculating dF/F that takes into account Suite2p neuropil
    estimations. Subtracting the scaled neuropil from the raw trace can be
    valuable for removing bleedthrough from nearby ROIs or other local diffuse
    signals that a low-pass baseline will obscure. However, this initial
    subtraction complicates estimating dF/F, as the resulting residual trace
    will have a baseline close to zero (or negative). Thus it is necessary to
    "add back in" the component of the low-pass baseline that was removed by
    the neuropil correction, to avoid distortions in the dF/F due to small or
    negative denominator terms.

    The unfiltered neuropil is first subtracted from the unfiltered raw trace.
    Two partial baselines are estimated then by filtering the residual raw
    trace, and the neuropil. The residual raw trace minus its partial baseline
    forms the numerator term of the dF/F. The denominator is the sum of the
    partial baselines.

    The partial baselines are calculated by applying a minimax filter to the
    respective (smoothed) traces.

    Parameters
    ----------
    window : int, optional
        Size of the rolling window for min/max/smoothing operations, in frames.
        Defaults to 600
    sigma : float
        Standard deviation of the gaussian smoothing kernel applied prior to
        min/max filtering, in frames. Defaults to 10.
    min_periods : float (0, 1], optional
        Fraction  of frames in each window that must be non-NaN in order to
        return a non-NaN baseline. Defaults to 0.2.
    constant_denominator : bool, optional
        Whether to divide the dF result by a constant denominator when
        calculating dF/F, taken as the median of the total baseline across
        all time points. This is helpful to avoid artifactually increasing
        signal amplitudes over time due to a monotonically decaying baseline,
        for example during photobleaching. Defaults to False

    Attributes
    ----------
    baseline : tuple
        Contains 3 DataFrames containing the total baseline ([0]),
        signal residual baseline ([1]), and neuropil baseline ([2]).
    """

    def __init__(self, window=600, sigma=10, min_periods=0.2,
                 constant_denominator=False):
        self.window = window
        self.sigma = sigma
        self.min_periods = min_periods
        self.constant_denominator = constant_denominator

    def _load_signals(self, experiment, channel, label):
        sig_dict = {
            'signal': experiment.signals(signal_type='raw', channel=channel,
                                         label=label, max_frame=None),
            'npil': experiment.signals(signal_type='npil', channel=channel,
                                       label=label, max_frame=None)
        }
        return sig_dict

    def _filter_final_signal(self, signal):
        """No final filtering"""
        return signal

    def _calculate_baseline(self, signal, npil):
        """doc string """

        sig_residual = signal - npil

        baseline_sig = filters.maxmin_filter(
            sig_residual, window=self.window, sigma=self.sigma,
            min_periods=self.min_periods)
        baseline_npil = filters.maxmin_filter(
            npil, window=self.window, sigma=self.sigma,
            min_periods=self.min_periods)

        baseline_total = baseline_sig + baseline_npil
        if self.constant_denominator:
            baseline_total = np.median(baseline_total)
        self.baseline = (baseline_total, baseline_sig, baseline_npil)

        return self.baseline

    def calculate(self, signal, npil):
        """Calculate dF/F

        Parameters
        ----------
        signal : array-like (n_rois, n_timepoints)
        npil : array-like (n_rois, n_timepoints)

        Returns
        -------
        dfof : pd.DataFrame
        """

        if len(signal.shape) != 2:
            raise ValueError("Input signal must be 2D")

        try:
            del self.baseline
        except AttributeError:
            pass

        base = self.calculate_baseline(signal, npil)
        return self._filter_final_signal(((signal - npil) - base[1]) / base[0])

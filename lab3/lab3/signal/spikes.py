"""Methods for spike inferencing. Implement custom strategies by subclassing
SpikesStrategy and filling in the details."""

from abc import abstractmethod
import warnings

import numpy as np
import pandas as pd
from multiprocessing import Pool
from oasis.functions import deconvolve  # should we move this into the class?

from lab3.core import Automorphism

warnings.filterwarnings("ignore", category=FutureWarning)


class SpikesStrategy(Automorphism):

    """Base class for spike inferencing strategies. Derived strategies should
    implement the _calculate_baseline and _do_inference methods."""

    name = "spikes"

    def apply_to(self, experiment, channel='Ch2', label=None, from_type='dfof',
                 **kwargs):
        """Apply spike inferencing analysis to an experiment object.

        Parameters
        ----------
        experiment : lab3.experiment.base.ImagingExperiment
            Instance of ImagingExperiment to analyze
        channel : str or int, optional
        label : str, optional
        from_type : str {'dfof', 'raw'} or custom, optional

        Additional keyword arguments are passed directly to `calculate`.
        """
        sig_dict = self._load_signals(experiment, channel=channel, label=label,
                                      from_type='dfof')
        return self.calculate(**sig_dict, **kwargs)

    def calculate_baseline(self, signals):
        try:
            return self.baseline
        except AttributeError:
            self.baseline = self._calculate_baseline(signals)
            return self.baseline

    def calculate(self, signals):
        """Run spike inferencing"""

        try:
            del self.baseline
        except AttributeError:
            pass
        baseline = self.calculate_baseline(signals)
        return self._do_inference(signals - baseline)

    # ----------------- Implement these methods to subclass ----------------- #

    @abstractmethod
    def _do_inference(self, signals):
        """Implement the spike inferencing algorithm here. It should return
        a dataframe the same size as signals, preserving the index"""

    # ------------------------------- Optional ------------------------------ #

    def _calculate_baseline(self, signals):
        """Estimate baseline for dFOF calculation. It should return a dataframe
        the same size as signals, preserving the index"""
        return np.zeros(signals.shape)

    def _load_signals(self, experiment, channel, label, from_type):
        """By default only a single signal dataframe is passed to `calculate`
        when using `apply`. If you override `calculate` to take additional
        arguments in your custom strategy, you should load and add them to the
        dictionary here, which will be unpacked as arguments to `calculate`.
        """
        sig_dict = {
            'signals': experiment.signals(signal_type=from_type,
                                          channel=channel, label=label,
                                          max_frame=None)
        }
        return sig_dict


class MADMixin:

    """Mixin that adds an additional spike filtering step following inference,
    by discarding spikes with amplitude less than some threshold number of
    median absolute deviations from the original trace. Since we override
    the calculate method, this should be positioned first in the inheritance
    list when subclassing.

    Derived classes should store an n_mad attribute, for calculating the
    thresholds, and expose this as a parameter to the user in the init
    method.
    """

    # TODO : should we just implement this using the min_spike in oasisAR1?

    def _mad_filter_spikes(self, signals, spikes):
        """Do MAD filtering"""

        mad = signals.sub(signals.median(axis=1), axis=0).abs().median(axis=1)
        self.thresholds = mad * self.n_mad
        spikes[spikes.le(self.thresholds, axis=0)] = 0
        return spikes

    def calculate(self, signals):
        """Calculates spikes and applies MAD filtering, if the n_mad attribute
        exists"""
        spikes = super().calculate(signals)

        try:
            return self._mad_filter_spikes(signals - self.baseline, spikes)
        except AttributeError:
            return spikes


def _deconvolve(inputs):
    """Must be a top level function for multiprocessing"""
    label, signal, g, kws = inputs
    return label, deconvolve(np.nan_to_num(signal), g=g, **kws)[1]


def _iter_signals(signals, g, kws):
    for label, signal in signals.iterrows():
        yield (label, signal, g, kws)


class OasisAR1Spikes(MADMixin, SpikesStrategy):

    """Infer spikes using the OASIS method, AR1 model variaton.

    TODO framerate should be in the metadata of the signals dataframe, so we
    don't have to pass it explicitly

    Parameters
    ----------
    tau : float, optional
        Decay time constant of the indicator, in seconds. Defaults to 0.7,
        which is a reasonable number for GCaMP6f.
    fs : float, optional
        Frame rate, in Hz. Defaults to 10.
    n_mad : int, optional
        Filter spikes following inference, discarding any events with an
        amplitude less than n_mad * median absolute deviation of the original
        signal trace (e.g., 3 n_mad corresponds roughly to discarding spikes
        with an amplitude less than 2 standard deviations of the original
        trace). Defaults to 0 (no filtering).
    n_processes : int, optional
    oasis_kws : dict, optional
        description

    Returns
    -------
    spikes : pd.DataFrame

    Attributes
    ----------
    baseline : pd.DataFrame
    thresholds : pd.Series
    """

    def __init__(self, tau=0.7, fs=10, n_mad=3, n_processes=1, oasis_kws={}):
        self.tau = tau
        self.fs = fs
        self.n_mad = n_mad
        self.oasis_kws = oasis_kws
        self.n_processes = n_processes

    @property
    def g(self):
        """Calculates g for the AR1 model"""
        return np.exp(-1 / (self.tau * self.fs))

    def _do_inference(self, signals):
        spikes = pd.DataFrame(np.zeros(signals.shape), index=signals.index)

        _signals = _iter_signals(signals, [self.g], self.oasis_kws)

        if self.n_processes > 1:
            p = Pool(self.n_processes)
            for l, s in p.map(_deconvolve, _signals):
                spikes.loc[l] = s

            p.close()
            p.join()

        else:
            for l, s in map(_deconvolve, _signals):
                spikes.loc[l] = s

        return spikes


class OasisAR2Spikes(MADMixin, SpikesStrategy):

    """Infer spikes using the OASIS method, AR1 model variaton.

    Parameters
    ----------
    tau_d, tau_r : float, optional
        Decay and rise time constant of the indicator, in seconds. Defaults to
        0.7 and 0.065 respectively, which are reasonable numbers for GCaMP6f.
    fs : float, optional
        Frame rate, in Hz. Defaults to 10.
    n_mad : int, optional
        description
    oasis_kws : dict, optional
        description

    Returns
    -------
    spikes : pd.DataFrame

    Attributes
    ----------
    baseline : pd.DataFrame
    thresholds : pd.Series
    """

    # TODO : Does this need to be a separate class? On the plus side it avoids
    # overloading tau? Though we could choose AR1 vs AR2 by whether tau_r
    # is passed?

    def __init__(self, tau_r=0.065, tau_d=0.7, fs=10, n_mad=3, oasis_kws={}):
        self.tau_r = tau_r
        self.tau_d = tau_d
        self.fs = fs
        self.n_mad = n_mad
        self.oasis_kws = oasis_kws

    @property
    def g(self):
        """Calculate g for the AR2 model"""
        term1 = np.exp(-1 / (self.tau_d * self.fs))
        term2 = np.exp(-1 / (self.tau_r * self.fs))
        return (term1 + term2, -1 * term1 * term2)

    def _do_inference(self, signals):

        from oasis.functions import deconvolve

        spikes = pd.DataFrame(np.zeros(signals.shape),
                              index=signals.index)
        for roi, signal in signals.iterrows():
            spikes.loc[roi] = deconvolve(signal.values, g=self.g,
                                         **self.oasis_kws)[1]

        return spikes

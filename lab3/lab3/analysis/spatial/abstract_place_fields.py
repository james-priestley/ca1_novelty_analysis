from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter1d
import scipy.stats as stats 
import pandas as pd
import itertools as it

from typing import TYPE_CHECKING

from lab3.core import Analysis
from lab3.core.filters import IndexIdentity, ColumnIdentity
from lab3.analysis.base import SignalAnalysis
from lab3.event import Event, EventGroup, LapEvent, LapGroup

from lab3.core.helpers import save_load

if TYPE_CHECKING:
    from lab3.experiment import ImagingExperiment

from . import SimpleSpatialTuning, SimplePFDetector, CircularSimpleSpatialTuning

def discretize_position(position, nbins=100):
    bins = np.linspace(0,1,nbins+1)[:-1]
    return np.searchsorted(bins, position) - 1

def discretize_laps(laps, nmax):
    out = np.ones(nmax)*(len(laps)-1)
    for i in range(1,len(laps)):
        out[laps[i-1]:laps[i]] = i-1

    return out

# TODO: Reconcile with abstract spatial tuning
class PlaceFields(SignalAnalysis):
    #name = 'PlaceFields'
    constructor_ids = ['num_bins', 'num_shuffles', 
                       'pval_threshold', 'min_trials_active', 'sigma']
    def to_columns(self):
        return pd.Index(['field_start', 'field_stop'])
    
    def __init__(self, num_shuffles=100, num_bins=100, pval_threshold=0.01, 
                 min_trials_active=3, sigma=3, **kwargs):
        """A shuffle-based place field identification algorithm. For each cell, 
        we draw surrogate tuning curves from a null distribution of shuffled
        signals. Locations where the true tuning curve exceeds some percentile 
        (default: 99th) of the shuffle distribution are identified as (statistically 
        significant) place fields.

        NB: Since this is a randomized (shuffle-based) algorithm, different runs may return 
        slightly different results.

        Initialization parameters
        ----------
        num_shuffles : int
            Number of shuffles to use to construct null distribution.
            The more the slower. Default: 100.
        num_bins : int
            Number of bins to discretize the track. Default: 100. 
        pval_threshold : float, optional
            Threshold to call a "significant" place field (default p<0.01)
        min_trials_active : int, optional
            Minimum number of laps a "place field" must occur on to be called a place field
        sigma : float, optional
            Standard deviation of the gaussian smoothing kernel applied to the
            tuning curves. By default, mild (3 bin) smoothing is applied.

        Examples
        --------
        >>> # Identify place fields
        >>> strategy = PlaceFields()
        >>> signal_spec = {"signal_type": "dfof", "label": "suite2p"}
        >>> place_fields = strategy.apply_to(
        >>>     expt, signal_spec=signal_spec)
        """

        super().__init__(num_shuffles=num_shuffles, num_bins=num_bins, 
                         pval_threshold=pval_threshold, 
                         min_trials_active=min_trials_active, sigma=sigma, 
                         **kwargs)
        self.num_shuffles = num_shuffles
        self.num_bins = num_bins
        self.pval_threshold=pval_threshold
        self.sigma = sigma
        self.min_trials_active = min_trials_active
        self.tuning_strategy = CircularSimpleSpatialTuning(sigma=sigma)
    
    @save_load
    def apply_to(self, expt : ImagingExperiment, signal_spec, roi_filter=IndexIdentity, 
                    time_filter=ColumnIdentity):
        laps = expt.format_behavior_data()['lap']
        position = expt.format_behavior_data()['treadmillPosition']
        signals = self._load_signals(expt, signal_spec, roi_filter=roi_filter, 
                                        time_filter=time_filter)
        pfs = self.calculate(signals, laps, position)
        pfs.columns = self.columns
        
        return pfs
    
    def calculate(self, signals, laps, position):
        discrete_position = discretize_position(position, nbins=self.num_bins)
        discrete_laps = discretize_laps(laps, len(discrete_position))

        if 0 < self.min_trials_active < 1.:
            min_trials = int(len(laps) * self.min_trials_active)
        else:
            min_trials = self.min_trials_active
        
        try:
            place_fields = SimplePFDetector(self.tuning_strategy, num_shuffles=self.num_shuffles, 
                                            pval_threshold=self.pval_threshold, 
                                            min_trials_active=min_trials).detect(
                                            signals, discrete_position, laps=discrete_laps)
        except:
            place_fields = None
        
        return place_fields


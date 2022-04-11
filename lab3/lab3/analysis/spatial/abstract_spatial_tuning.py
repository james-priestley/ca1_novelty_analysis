from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter1d
import scipy.stats as stats 
import pandas as pd
import itertools as it

from typing import TYPE_CHECKING

from lab3.core import Analysis
from lab3.core.filters import IndexIdentity, ColumnIdentity
from lab3.core.helpers import save_load
from lab3.analysis.base import SignalAnalysis
from lab3.event import Event, EventGroup, LapEvent, LapGroup

from lab3.core.helpers import save_load

if TYPE_CHECKING:
    from lab3.experiment import ImagingExperiment

class _SingleLapSpatialTuning(SignalAnalysis):
    name = 'SpatialTuning'
    constructor_ids = ['nbins', 'bins', 'sigma']
    def to_columns(self):
        return pd.Index(self.get_bins(nbins=self.nbins, bins=self.bins), name='position')
    
    def __init__(self, nbins: int = 20, bins=None, sigma: float = 3, 
                normalize: bool = True, save_results=True, load_saved=True, verbose=False):
        self.bins = self.get_bins(nbins, bins)
        self.nbins = len(self.bins)
        self.sigma = sigma
        self.normalize = normalize
        self.save_results = save_results
        self.load_saved = load_saved
        self.verbose = verbose
        self.loaded = False
        
    def apply_to(self, lap : LapEvent, signals, positions):
        onset = int(lap.onset)
        offset = int(lap.offset)
        return self._calculate(signals, positions, onset, offset)
    
    def _calculate(self, signals, positions, onset, offset):
        lap_signals = signals.iloc[:, onset:offset]
        lap_positions = positions[onset:offset]

        lap_signals.columns = self._to_bin(lap_positions)
        if self.normalize:
            tuning_curves = lap_signals.T.groupby('position').mean().T
        else:
            tuning_curves = lap_signals.T.groupby('position').sum().T
        
        if self.sigma is not None:
            tuning_curves = self._smooth(tuning_curves)
        
        return tuning_curves
    
    def _smooth(self, tuning_curves):        
        # This is an inplace operation
        gaussian_filter1d(tuning_curves, self.sigma, axis=1, 
        					output=tuning_curves, mode='wrap')
        return tuning_curves
    
    def _to_bin(self, x):
        return self.columns[np.searchsorted(self.bins, x, side='right') - 1]
    
    @classmethod
    def get_bins(cls, nbins=None, bins=None):
        if nbins is not None:
            bins = np.linspace(0,1,nbins+1)[:-1]
        elif nbins is not None:
            bins = np.sort(bins)
            nbins = len(bins)
        else:
            raise ValueError("One of `bins` and `nbins` must be specified!")
            
        return bins
    
class SpatialTuning(_SingleLapSpatialTuning):
    """A simple spatial tuning strategy. For each neuron, we compute its
    average activity across samples/laps for each position. The tuning curves
    are optionally smoothed with a Gaussian kernel. Since we are computing
    the average activity across samples for a given position, this implicitly
    normalizes for occupancy.

    Initialization parameters
    ----------
    nbins : int
        Number of bins to discretize the track
    sigma : float, optional
        Standard deviation of the gaussian smoothing kernel applied to the
        tuning curves. By default, no smoothing is applied.

    Examples
    --------
    >>> # Compute the tuning curves
    >>> strategy = SpatialTuning(sigma=2)
    >>> signal_spec = {"signal_type": "dfof", "label": "suite2p"}
    >>> tuning_curves = strategy.apply_to(
    >>>     expt, signal_spec=signal_spec)
    """

    @save_load
    def apply_to(self, expt, signal_spec, roi_filter=IndexIdentity(), 
                    time_filter=ColumnIdentity(), verbose=False, behavior_spec={}):
        signals = self._load_signals(expt, signal_spec)
        signals = signals.loc[roi_filter.bind_to(expt)]

        # Because of laps and positions, these cannot simply be 
        # stripped out
        # TODO: Is replacing with nan ALWAYS preferable?
        signals.loc[:, ~(time_filter.bind_to(expt))] = np.nan

        laps = LapGroup.from_expt(expt)
        positions = expt.format_behavior_data(**behavior_spec)['treadmillPosition']

        return self.calculate(signals, laps, positions)
 
    def calculate(self, signals, laps, positions):
        return laps.apply(super(), signals=signals, positions=positions)


class EventSpatialTuning(SpatialTuning):
    
    def apply_to(self, expt, event_spec, **kwargs):
        events = expt.events(**event_spec)
        index  = expt.signals().index
        
        positions = expt.format_behavior_data()['treadmillPosition']
        imaging_times = expt.imaging_times()
        
        return self.calculate(events, positions, imaging_times, index)
    
    def calculate(self, events, positions, imaging_times, index):
        def _to_position(onset):
            idx = np.searchsorted(imaging_times, onset) - 1
            return positions[idx]
                
        results = pd.DataFrame(np.zeros((len(index),self.nbins)), 
                               index=index, 
                               columns=self.columns)
        
        bins = events.onset.map(lambda x: self._to_bin(_to_position(x)))
        counts = bins.value_counts()
        results.loc[:, counts.index] += counts
        
        return results

class Occupancy(SpatialTuning):
    """Calculates the amount of time spent in each bin
    """
    name = 'Occupancy'
    def apply_to(self, expt, *args, **kwargs):
        occupancy_normalize = self.normalize
        self.normalize = False 

        result = super().apply_to(expt, *args, **kwargs)
        result.index = result.index.droplevel(None)

        if occupancy_normalize:
            result /= result.sum()

        return result

    def _load_signals(self, expt, signal_spec):
        signals = super()._load_signals(expt, signal_spec)
        signals = pd.DataFrame(np.ones((1,signals.shape[1])), 
                               columns=signals.columns)
        return signals

class PopulationVectorCorrelations(SpatialTuning):
    constructor_ids = ['nbins', 'bins', 'sigma', 'correlation_function']
    
    def to_columns(self):
        return pd.Index(['correlation', 'pval'])
    
    def __init__(self, correlation_function=stats.pearsonr, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.spatial_tuning = SpatialTuning(*args, **kwargs)
        self.correlation_function = correlation_function
        
    def apply_to(self, expt : ImagingExperiment, signal_spec, 
                     roi_filter=IndexIdentity(), 
                     time_filter=ColumnIdentity(), time_filter2=None, 
                 **corr_kwargs):
        tuning_curves = self.spatial_tuning.apply_to(expt, signal_spec=signal_spec, 
                                                     roi_filter=roi_filter, 
                                                     time_filter=time_filter)
        tuning_curves = tuning_curves.groupby('roi_label').agg('mean')
        
        if time_filter2 is not None:
            tuning_curves2 = self.spatial_tuning.apply_to(expt, signal_spec=signal_spec, 
                                                         roi_filter=roi_filter, 
                                                         time_filter=time_filter2)
            tuning_curves2 = tuning_curves2.groupby('roi_label').agg('mean')            
        else:
            tuning_curves2 = None
        
        return self.calculate(tuning_curves, tuning_curves2, **corr_kwargs)           
        
    def calculate(self, tuning_curves, tuning_curves2=None, **corr_kwargs):

        if tuning_curves2 is None:
            tuning_curves2 = tuning_curves
        
        def corr(series):
            x = tuning_curves.loc[:, series.bin_1] 
            y = tuning_curves2.loc[:, series.bin_2]

            is_na = pd.isnull(x) + pd.isnull(y)

            try:
                return self.correlation_function(x.loc[~is_na], 
                                             y.loc[~is_na], 
                                             **corr_kwargs)
            except:
                return np.nan
        
        pairs = pd.DataFrame(it.product(tuning_curves.columns, 
                                        tuning_curves2.columns), 
                             columns=['bin_1', 'bin_2'])
        
        correlation_tuples = pairs.apply(corr, axis=1)
        
        result = pd.DataFrame([tup for tup in correlation_tuples],
                              index=pd.MultiIndex.from_frame(pairs),
                              columns=self.columns)
        
        return result
    
class TuningCurveCorrelations(SpatialTuning):
    constructor_ids = ['nbins', 'bins', 'sigma', 'correlation_function']
    
    def to_columns(self):
        return pd.Index(['correlation', 'pval'])
    
    def __init__(self, correlation_function=stats.pearsonr, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.spatial_tuning = SpatialTuning(*args, **kwargs)
        self.correlation_function = correlation_function
        
    def apply_to(self, expt : ImagingExperiment, signal_spec, 
                     roi_filter=IndexIdentity(), 
                     time_filter=ColumnIdentity(), time_filter2=None, 
                     **corr_kwargs):
        tuning_curves = self.spatial_tuning.apply_to(expt, signal_spec=signal_spec, 
                                                     roi_filter=roi_filter, 
                                                     time_filter=time_filter)
        tuning_curves = tuning_curves.groupby('roi_label').agg('mean')
        
        if time_filter2 is not None:
            tuning_curves2 = self.spatial_tuning.apply_to(expt, signal_spec=signal_spec, 
                                                         roi_filter=roi_filter, 
                                                         time_filter=time_filter2)
            tuning_curves2 = tuning_curves2.groupby('roi_label').agg('mean')            
        else:
            tuning_curves2 = None
        
        return self.calculate(tuning_curves, tuning_curves2, **corr_kwargs)        
        
    def calculate(self, tuning_curves, tuning_curves2=None, **corr_kwargs):
        
        if tuning_curves2 is None:
            tuning_curves2 = tuning_curves
        
        def corr(series):
            x = tuning_curves.loc[series.cell_1] 
            y = tuning_curves2.loc[series.cell_2]
            
            is_na = pd.isnull(x) + pd.isnull(y)
            
            return self.correlation_function(x.loc[~is_na], 
                                             y.loc[~is_na], 
                                             **corr_kwargs)
    def calculate(self, tuning_curves, tuning_curves2=None, **corr_kwargs):
        
        if tuning_curves2 is None:
            tuning_curves2 = tuning_curves
        
        def corr(series):
            x = tuning_curves.loc[series.cell_1] 
            y = tuning_curves2.loc[series.cell_2]
            
            is_na = pd.isnull(x) + pd.isnull(y)
            
            return self.correlation_function(x.loc[~is_na], 
                                             y.loc[~is_na], 
                                             **corr_kwargs)
        
        pairs = pd.DataFrame(it.product(tuning_curves.index, 
                                        tuning_curves2.index), 
                             columns=['cell_1', 'cell_2'])
        
        correlation_tuples = pairs.apply(corr, axis=1)
        
        result = pd.DataFrame([tup for tup in correlation_tuples],
                              index=pd.MultiIndex.from_frame(pairs),
                              columns=self.columns)
        
        return result

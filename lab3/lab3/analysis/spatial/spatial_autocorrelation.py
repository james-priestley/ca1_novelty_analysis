import numpy as np
import pandas as pd
from scipy.ndimage import correlate1d

from .abstract_spatial_tuning import SpatialTuning

def _cyclic_autocorr(array, mode='wrap', pearson=True):
    output = np.zeros(len(array) + 1, dtype=float)
    
    if pearson:
        array -= np.mean(array)
        array /= np.std(array)*np.sqrt(len(array))
    
    correlate1d(array, array, mode=mode, output=output[:-1], origin=0)
    output[-1] = output[0]
    
    return output

class AutocorrelationMixin:
    
    constructor_ids = ['max_lag']
    
    def to_columns(self):
        return pd.RangeIndex(-self.max_lag, self.max_lag+1)
    
    def calculate(self, dataframe, **acorr_kwargs):
        autocorrs = dataframe.apply(_cyclic_autocorr, **acorr_kwargs, axis=1)
        autocorrs = autocorrs.apply(pd.Series)
        autocorrs.columns = self.columns
        return autocorrs        

class SpatialAutocorrelation(AutocorrelationMixin, SpatialTuning):
    
    constructor_ids = SpatialTuning.constructor_ids + AutocorrelationMixin.constructor_ids
    
    def __init__(self, **spatial_tuning_kwargs):
        self.spatial_tuning = SpatialTuning(**spatial_tuning_kwargs)
        super().__init__(**spatial_tuning_kwargs)
        self.max_lag = self.nbins//2
    
    def apply_to(self, expt, signal_spec):
        tuning_curves = self.spatial_tuning.apply_to(expt, signal_spec=signal_spec)
        return self.calculate(tuning_curves)
    
class SpatialAutocorrelationWidth(SpatialAutocorrelation):
    
    constructor_ids = SpatialAutocorrelation.constructor_ids + ['cutoff']
    
    def __init__(self, cutoff, **spatial_tuning_kwargs):
        self.spatial_autocorrelation = SpatialAutocorrelation(**spatial_tuning_kwargs)
        self.cutoff = cutoff
        super().__init__(**spatial_tuning_kwargs)
        
    def to_columns(self):
        return pd.Index(["width"])
    
    def apply_to(self, expt, signal_spec, groupby=None, agg=None):
        spatial_autocorrelation = self.spatial_autocorrelation.apply_to(expt, signal_spec=signal_spec)
        
        if groupby is not None and agg is not None:
            spatial_autocorrelation = spatial_autocorrelation.groupby(groupby).agg(agg)
            
        return self.calculate(spatial_autocorrelation)
        
    def calculate(self, spatial_autocorrelation):
        def get_width(series):
            if np.any(np.isnan(series)):
                return np.nan
            else:
                return 2*np.argmax(series.loc[0:] < self.cutoff)
        
        widths = spatial_autocorrelation.apply(get_width, axis=1)
        widths = widths.to_frame()
        widths.columns = self.columns
        return widths

import numpy as np
import pandas as pd

from lab3.core import Analysis
from lab3.filters import IndexIdentity, ColumnIdentity

class Histogram(Analysis):
    constructor_ids = ['lower', 'upper', 'nbins', 'bins']
    
    def __init__(self, lower=0., upper=1., nbins=100, bins=None):
        if bins is None:
            self.lower = lower
            self.upper = upper
            self.nbins = nbins
            self.bins = np.linspace(lower, upper, nbins+1)
        else:
            self.lower = bins[0]
            self.upper = bins[-1]
            self.nbins = len(bins)
            self.bins = bins
            
        self.bin_size = self.bins[1] - self.bins[0]
    
    def to_columns(self):
        return pd.Index(self.bins[:-1], name=self.name)
    
    def apply_to(self, expt, signal_spec, roi_filter=IndexIdentity, time_filter=ColumnIdentity):
        signals = expt.signals(**signal_spec)
        signals = signals.loc[roi_filter.bind_to(expt)]
        signals.loc[:, ~time_filter.bind_to(expt)] = np.nan
        return self.calculate(signals)
        
    def calculate(self, signals):
        clean_signals = signals.dropna(axis=1)
                
        result = clean_signals.apply(lambda x: np.histogram(x, bins=self.bins)[0], axis=1)
        return pd.DataFrame((item for item in result), index=result.index, columns=self.columns)

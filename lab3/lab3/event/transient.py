import pandas as pd 

from lab3.core.classes import Group
from . import IntervalEvent, EventGroup, EventsFile
from .group import Group2DataFrameMixin, ImagingEventGroupGroup       

class Transient(IntervalEvent):
    constructor_ids = ['onset', 'offset', 'sigma', 'duration', 
                       'amplitude', 'amplitude_index']
    #def __init__(self, onset, offset, sigma, duration, amplitude, amplitude_index):
    #    self.onset = onset
    #    self.offset = offset
    #    self.sigma = sigma
    #    self.duration = duration
    #    self.amplitude = amplitude
    #    self.amplitude_index = amplitude_index
    
class SingleCellTransients(EventGroup):
    item_class = Transient

class ExperimentTransientGroup(ImagingEventGroupGroup):
    item_class = SingleCellTransients

    def __getitem__(self, key):
        try:
            return super().__getitem__(key)
        except KeyError:
            for single_cell_transients in self:
                if single_cell_transients.name == key:
                    return single_cell_transients
                
    @classmethod
    def from_pandas(cls, dataframe, trigger=None, **kwargs):
        return cls([SingleCellTransients.from_pandas(df, name=roi_label, trigger=trigger) 
                    for roi_label, df in dataframe.groupby("roi_label")])



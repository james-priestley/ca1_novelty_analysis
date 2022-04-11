"""Common metrics calculate from spatial tuning curves and place fields"""

import numpy as np
import pandas as pd
from astropy.stats import kuiper_two

from lab3.core import Analysis
from .abstract_spatial_tuning import EventSpatialTuning, Occupancy

from lab3.core.helpers import save_load

# Should these generically take single or multiple tuning curves/place fields?
# Probably just single, and use pandas to apply row-wise on the corresponding
# dataframes


class SkaggsSpatialInformation(EventSpatialTuning):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.spatial_tuning_strategy = EventSpatialTuning(*args, **kwargs)
        self.occupancy_strategy = Occupancy(*args, **kwargs)
    
    def apply_to(self, expt, event_spec, **kwargs):
        event_tuning = self.spatial_tuning_strategy.apply_to(expt, event_spec=event_spec, **kwargs)
        occupancy = self.occupancy_strategy.apply_to(expt, signal_spec={}, **kwargs).sum(axis=0)
        
        return self.calculate(event_tuning, occupancy)
    
    
    def calculate(self, event_tuning, occupancy):
        norm_occupancy = occupancy / occupancy.sum()
        
        ratio = event_tuning.div(event_tuning.mean(axis=1), axis=0)
        log2ratio = np.nan_to_num(np.log2(ratio))
        
        return norm_occupancy * ratio * log2ratio

class Kuiper(Analysis):
    def to_columns(self):
        return pd.Index(['statistic', 'p-value'])
    
    @save_load
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
        
        event_positions = events.onset.map(_to_position)
        
        try:
            kuiper_result = kuiper_two(event_positions, positions)
        except:
            kuiper_result = (np.nan, np.nan)
        
        result = pd.DataFrame([kuiper_result], 
                              index=index, columns=self.columns)
        
        return result

def spatial_reliability():
    return np.nan


def spatial_tuning_gain():
    raise NotImplementedError


def place_field_gain():
    raise NotImplementedError


def first_lap():
    """Return the lap number for the first time the cell fired in its place
    field"""
    return np.nan

from __future__ import annotations

import pandas as pd
import numpy as np
import copy
import scipy.signal as ss
import warnings

# Type annotations without circular import
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from lab3.experiment import BehaviorExperiment

from lab3.core import Item, Group

from .base import EventsFile, Event, OnsetEvent, OffsetEvent, IntervalEvent, bool2events
from .group import BehaviorEventGroup

class SpatialEvent(Event):
    def __init__(self, *args, **kwargs):
        """
        
        TO DOCUMENT

        """
        super().__init__(*args, **kwargs)

        if self.duration > 0.5:
            warnings.warn(f"Creating a SpatialEvent that spans {self.duration*100:.0f}% the belt, "
                    "are you sure this is what you want to do?")

    def to_bin(self, bins):
        bin_on = np.searchsorted(bins, self.onset, side='right') - 1
        if np.isnan(self.offset):
            bin_off = np.nan
        else:
            bin_off = np.searchsorted(bins, self.offset, side='right') - 1
            
        return self.__class__(onset=bin_on, offset=bin_off)
    
    def to_time_events(self, expt, name=None):
        if name is None:
            name = f"position_({self.onset}, {self.offset})"
        
        positions = expt.format_behavior_data()['treadmillPosition']
        bool_array = (self.onset < positions) & (positions < self.offset)
        return BehaviorEventGroup(bool2events(bool_array), name=name)

class SpatialEventGroup(BehaviorEventGroup):
    @classmethod
    def load(cls, expt : BehaviorExperiment, behavior_key : str, **kwargs):
        events = super().load(expt, behavior_key, as_time_events=False, **kwargs)
        positions = pd.Series(expt.format_behavior_data()['treadmillPosition'], name='positions')
        onsets = positions.loc[events.onset]
        offsets = positions.loc[events.offset]
        
        return cls([SpatialEvent(onset=on, offset=off) 
                    for on, off in zip(onsets, offsets)], name=behavior_key)
    
    def positions2bins(self, nbins=None, bins=None):
        assert bins or nbins
        if bins is None:
            bins = np.linspace(0, 1, nbins)
        
        return self.__class__([event.to_bin(bins) for event in self], 
                              name=self.name)
    
    def to_time_events(self, expt):
        nframes = len(expt.imaging_times())
        bool_array = np.sum([spatial_event.to_time_events(expt).bool_array(nframes) 
                             for spatial_event in self], axis=0)
        
        return BehaviorEventGroup(bool2events(bool_array), name=self.name)
        

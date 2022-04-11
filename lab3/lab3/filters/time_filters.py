from __future__ import annotations
import pandas as pd 
import numpy as np
from functools import reduce
from typing import TYPE_CHECKING

from lab3.core.filters import TimeFilter, _BoundFilter
from lab3.event import SpatialEvent

if TYPE_CHECKING:
    from lab3.experiment import ImagingExperiment

class SpatialFilter(TimeFilter, SpatialEvent):
    """Select timepoints where the animal was in a spatial location
    
    SpatialFilters can be combined with the usual Filter grammar.
    """
    def bind_to(self, expt : ImagingExperiment):
        events = super().to_time_events(expt)
        idxes = [pd.RangeIndex(ev.onset, ev.offset) for ev in events]
        return self.bind(reduce(pd.Index.union, idxes))

class EventFilter(TimeFilter):
    """Selects activity inside events, specified with a dictionary.
    Example
    -------
    >>> event_spec = {
            'signal_type': 'behavior',
            'event_type': 'running',
            'as_time_events': False
            }
    >>> running_filter = EventFilter(event_spec)
    >>> is_running = running_filter.bind_to(expt)
    >>> is_not_running = ~is_running

    """
    constructor_ids = ['event_spec']
    def __init__(self, event_spec):
        self.event_spec = event_spec

    def bind_to(self, expt : ImagingExperiment):
        events = expt.events(**self.event_spec)
        idxes = [pd.RangeIndex(ev.onset, ev.offset) for ev in events]
        return self.bind(reduce(pd.Int64Index.union, idxes, pd.Int64Index([])))

class IsRunning(EventFilter):
    constructor_ids = ['threshold', 'debounce_window', 'as_time_events']
    def __init__(self, threshold=4, debounce_window=4, as_time_events=False):
        self.threshold = threshold
        self.debounce_window = debounce_window
        self.as_time_events = as_time_events
        
        event_spec = {
            'signal_type': 'behavior', 
            'event_type': 'running',
            'fmt': 'interval',
            'threshold': threshold,
            'debounce_window': debounce_window,
            'as_time_events': as_time_events
        }
        
        super().__init__(event_spec=event_spec)

class InReward(EventFilter):
    constructor_ids = []
    def __init__(self):
        event_spec = {
            'signal_type': 'behavior', 
            'event_type': 'reward',
            'fmt': 'interval',
            'as_time_events': False            
        }
        
        super().__init__(event_spec=event_spec)

class RealTimeEventFilter(EventFilter):
    constructor_ids = ['event_spec', 'pre', 'post']
    def __init__(self, event_spec, pre=0., post=0.):
        self.pre = pre
        self.post = post

        super().__init__(event_spec=event_spec)


    def bind_to(self, expt):

        events = expt.events(**self.event_spec)
        imaging_times = expt.imaging_times()

        idxes = []
        for ev in events:
            lower, = np.where(imaging_times >= ev.onset - self.pre)
            upper, = np.where(imaging_times <= ev.offset + self.post)
            idxes.append(pd.RangeIndex(lower[0], upper[-1]))

        return self.bind(reduce(pd.Int64Index.union, idxes, pd.Int64Index([])))



    

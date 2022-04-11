import numpy as np 
import pandas as pd 

from lab3.metadata import DefaultMetadata

class SignalGetter:
    def __call__(self, expt, **kwargs):
        raise NotImplementedError

class Event2Signals(SignalGetter):
    def __init__(self, event_spec, signal_type, 
            groupby='roi_label', metadata=DefaultMetadata): 
        self.event_spec = event_spec
        self.groupby = groupby
        self.signal_type = signal_type
        self.metadata = metadata
    
    def __call__(self, expt, **kwargs):
        raise NotImplementedError
    
class EventOnsets2Signals(Event2Signals):
    def __call__(self, expt, **kwargs):
        event_df = expt.events(**self.event_spec).dataframe
        
        signals = expt.signals(signal_type=self.signal_type, 
                    channel=self.event_spec['channel'], metadata=self.metadata)
        signals.loc[:,:] = 0
        
        for roi_label, events in event_df.groupby(self.groupby):
            onsets = events.onset.astype(int)
            idx = signals.level[self.groupby].get_loc(roi_label)
            signals.iloc[idx, onsets] = 1.
            
        return signals

class TimeOnsets2Signals(EventOnsets2Signals):
    def __call__(self, expt, **kwargs):
        imaging_times = expt.imaging_times()
        event_df = expt.events(**self.event_spec).dataframe
        
        signals = expt.signals(signal_type=self.signal_type, 
                    channel=self.event_spec['channel'], metadata=self.metadata)
        signals.loc[:,:] = 0
        
        for roi_label, events in event_df.groupby(self.groupby):
            onsets = events.onset
            frame_idxes = np.searchsorted(imaging_times, onsets) + 1.
            idx = signals.level[self.groupby].get_loc(roi_label)
            signals.iloc[idx, frame_idxes.astype(int)] = 1.
            
        return signals
        
        
        

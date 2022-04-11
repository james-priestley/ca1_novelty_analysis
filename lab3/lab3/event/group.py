import pandas as pd
import numpy as np
import copy
import scipy.signal as ss

from scipy.ndimage.filters import median_filter

from lab3.core import Item, Group
import lab3.experiment.base as leb

from .base import EventsFile, Event, OnsetEvent, OffsetEvent, IntervalEvent, bool2events

class Group2DataFrameMixin:
    @classmethod
    def load(cls, events_file_path, event_type, channel=None, label=None, trigger=None):
        # Handle object of type `Experiment` passed
        try:
            events_file_path = events_file_path.imaging_events_path
        except AttributeError:
            pass

        with EventsFile(events_file_path) as events_file:
            events = events_file.get(f"{label}/{event_type}/{channel}")

        return cls.from_pandas(events, name=event_type, trigger=trigger)

    def save(self, events_file_path, event_type, channel=None, 
             label=None):
        with EventsFile(events_file_path) as events_file:
            events_file.put(f"{label}/{event_type}/{channel}", self.dataframe)        

    @classmethod
    def from_pandas(cls, dataframe, name='unnamed', item_class=None, **kwargs):
        if item_class is None:
            item_class = cls.item_class
        return cls([item_class(**row, **kwargs) for _, row in dataframe.iterrows()], 
                    name=name, item_class=item_class)

    def to_pandas(self):
        return pd.concat([ev.to_pandas() for ev in self],
                keys=pd.RangeIndex(len(self)), 
                names=[self.item_class.name]).unstack()        

    @property
    def dataframe(self):
        return self.to_pandas()        

class EventGroup(Group, Group2DataFrameMixin):

    item_class = Event
    
    @property
    def onset(self):
        return pd.Series([event.onset for event in self], name='onset')
    
    @property
    def offset(self):
        return pd.Series([event.offset for event in self], name='offset')
    
    def expand(self, pre=0, post=0):
        df = self.dataframe
        df['onset'] -= pre
        df['offset'] += post
        return self.from_pandas(df)

    def frames2times(self, expt):
        new_events = []
        imaging_times = pd.Series(expt.imaging_times())
        for event in self:
            ids = {
                key:imaging_times.loc[value] 
                    for key, value in event.constructors().items()
            }
            new_events.append(self.item_class(**ids))

        return self.__class__(new_events)
    
    def bool_array(self, n_frames):
        bool_array = np.zeros(n_frames)
        idx_array = np.arange(n_frames)
        for event in self:
            bool_array[(idx_array < event.offset) & (event.onset <= idx_array)] = 1

        return bool_array.astype(bool)
    
    def __contains__(self, time):
        return np.any([time in event for event in self])

    def filter_trials(self, trial_filter):
        return self.from_pandas(self.dataframe.loc[trial_filter], name=self.name, 
                                item_class=self.item_class)

class LFPEventGroup(EventGroup):
    item_name = 'LFPEvent'
    @classmethod
    def load(cls, expt, event_type, label='LFP', channel=None):
        return super().load(expt.lfp_events_path, event_type, 
                                label=label, channel=channel)
    
class BehaviorEventGroup(EventGroup):
    item_name = 'BehaviorEvent'
    @classmethod
    def load(cls, expt, behavior_key, as_time_events=False, **kwargs):
        if behavior_key == 'running':
            events = cls._load_running(expt, **kwargs)
        elif behavior_key == 'lap':
            events = cls._load_lap(expt, **kwargs)
        elif behavior_key == 'position':
            events = cls._load_position(expt, **kwargs)
        else:
            events = cls._load_generic(expt, behavior_key, **kwargs)

        if as_time_events:
            return events.frames2times(expt)
        else:
            return events

    @classmethod
    def _load_running(cls, expt, threshold=4, debounce_window=4, fmt='interval'):
        behavior_data = expt.format_behavior_data()
        velocity = behavior_data['velocity'].astype(int)
        # window must be odd
        try:
            window = 2*(int(expt.frame_rate*debounce_window)//2)+1
        except AttributeError:
            window = 2*(int(10*debounce_window)//2)+1
        running_array = median_filter(velocity > threshold, window).astype(int)

        if fmt == 'offset':
            events = cls(bool2events(running_array, event_class=OffsetEvent), name='running_start')
        elif fmt == 'onset':
            events = cls(bool2events(running_array, event_class=OnsetEvent), name='running_stop')
        else:
            events = cls(bool2events(running_array, event_class=IntervalEvent), name='running')

        return events

    @classmethod
    def _load_position(cls, expt, position):
        behavior_data = expt.format_behavior_data()
        position_array = behavior_data['treadmillPosition'] - position
        onsets, = np.where(np.diff(np.signbit(position_array)))

        events = cls([OnsetEvent(trigger) for trigger in onsets], 
                    name=f"position_{position:0.3f}")

        return events

    @classmethod 
    def _load_lap(cls, expt, **kwargs):
        from .lap import LapGroup
        return LapGroup.from_expt(expt)

    @classmethod
    def _load_generic(cls, expt, behavior_key, fmt='interval'):
        behavior_data = expt.format_behavior_data()
        behavior_array = behavior_data[behavior_key].astype(int)
        
        if fmt == 'offset':
            events = cls(bool2events(behavior_array, event_class=OffsetEvent), name=behavior_key)
        if fmt == 'onset':
            events = cls(bool2events(behavior_array, event_class=OnsetEvent), name=behavior_key)
        else:
            events = cls(bool2events(behavior_array, event_class=IntervalEvent), name=behavior_key)

        return events

class EventGroupGroup(Group):
    item_class = EventGroup

    def to_pandas(self):
        return pd.concat([grp.dataframe for grp in self],
                keys=[grp.name for grp in self],
                names=[self.item_class.name])

    @property
    def dataframe(self):
        return self.to_pandas()

class ImagingEventGroupGroup(EventGroupGroup, Group2DataFrameMixin):
    def to_pandas(self):
        return pd.concat([grp.dataframe for grp in self],
                keys=[grp.name for grp in self],
                names=['roi_label'])


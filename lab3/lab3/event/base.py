import pandas as pd
import numpy as np
import copy
import scipy.signal as ss

from lab3.core import Item, Group

class EventsFile(pd.HDFStore):
    pass

class Event(Item):
    name = 'Event'
    constructor_ids = ['onset', 'offset']

    def __init__(self, onset=None, offset=None, trigger=None, **kwargs):
        self.onset = onset 
        self.offset = offset
        self.trigger_key = trigger

        super().__init__(**kwargs)

    def label(self):
        return str((self.onset, self.offset))

    def validate(self):
        """Check that this event is valid. Useful if writing to events.h5
        More important for subclasses of Event
        """
        assert 'onset' in self
        # assert self.duration > 0
        self.point_estimate()
        
    @property
    def duration(self):
        try:
            return self.offset - self.onset
        except (AttributeError, TypeError):
            return 0
    
    @property
    def trigger(self):
        """Localize the event to a single point in time.
        """
        try:
            return self.__dict__[self.trigger_key]
        except:
            return self.onset

    def to_frame(self):
        """Localize the event to a frame.
        This needs to be overridden for non-imaging events
        """
        return self.trigger 

    @classmethod
    def from_pandas(cls, series, **kwargs):
        return cls(**series, **kwargs)

    def to_pandas(self):
        return pd.Series(self.constructors(), name=self.name)

    def __getitem__(self, key):
        return self.__dict__[key]
    
    def __contains__(self, time):
        return self.onset <= time < self.offset 

class TriggerEvent(Event):
    constructor_ids = ['onset']

    def __init__(self, onset):
        super().__init__(onset=onset, offset=onset)

    def label(self):
        return self.onset

class OnsetEvent(TriggerEvent):
    pass

class OffsetEvent(TriggerEvent):
    @property
    def trigger(self):
        return self.offset

class IntervalEvent(Event):
    pass

def bool2events(array, event_class=Event):
    events = []
    
    i = 0
    in_event = False
    onset = None
    offset = None
    while i < len(array)-1:
        if array[i] and not in_event:
            onset = i
            in_event = True
        elif not array[i] and in_event:
            offset = i
            in_event = False            
            events.append(event_class(onset=onset, offset=i))
            onset = None
            offset = None
        i += 1
    if in_event:
        events.append(event_class(onset=onset, offset=i))
    return events
        
    
    
    
        
    

from __future__ import annotations

import numpy as np
from typing import TYPE_CHECKING

from .base import Event
from .group import BehaviorEventGroup

if TYPE_CHECKING:
    from lab3.experiment import BehaviorExperiment

class LapEvent(Event):
    constructor_ids = Event.constructor_ids + ['lap_num'] 

    def __init__(self, lap_num, **event_kwargs):
        self.lap_num = lap_num
        super().__init__(**event_kwargs)

    def label(self):
        return self.lap_num

class LapGroup(BehaviorEventGroup):
    @classmethod
    def from_expt(cls, expt : BehaviorExperiment):
        try:
            lap_starts = expt.format_behavior_data()['lap']
        except KeyError:
            print(f"Handling missing laps for trial {expt.trial_id}! "
                     f"(Maybe look into why this happens?)")
            lap_starts, = np.where(np.diff(expt.format_behavior_data()['treadmillPosition']) < -0.75)
            lap_starts = np.concatenate([np.array([0]), lap_starts])
        nframes = expt.imaging_dataset.num_frames
        laps = [LapEvent(lap_num=i, onset=lap_starts[i], offset=lap_starts[i+1]) 
                    for i in range(len(lap_starts)-1)]
        laps += [LapEvent(lap_num=len(lap_starts)-1, onset=lap_starts[-1], offset=nframes)]
        return cls(laps)


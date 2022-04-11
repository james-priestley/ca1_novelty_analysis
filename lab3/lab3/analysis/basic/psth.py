from __future__ import annotations
import numpy as np
import pandas as pd

from typing import Dict, TYPE_CHECKING

from lab3.analysis.base import SignalAnalysis
from lab3.core import Analysis
from lab3.core.filters import IndexIdentity, ColumnIdentity

from lab3.core.helpers import save_load

if TYPE_CHECKING:
    from lab3.experiment import ImagingMixin

class _SingleEventPSTH(SignalAnalysis):

    name = 'PSTH'
    constructor_ids = ['pre', 'post', 'dt', 'agg']

    def __init__(self, pre: float = 3., post: float = 3., dt: float = 0.03, agg: str = 'sum'):
        self.pre = pre
        self.post = post
        self.dt = dt
        self.agg = agg

    def to_columns(self):
        return pd.TimedeltaIndex(np.arange(-self.pre, self.post+self.dt, self.dt), 
                                    name=self.name, unit='s')

    
    def apply_to(self, event, signals):
        psth = self._calculate_single_event(event.trigger, 
                                            signals, 
                                            pre=self.pre, 
                                            post=self.post) 
        return self._align_time(psth)

    def _calculate_single_event(self, trigger, signals, pre, post):

        # WHY CAN'T THIS BE NAMED self.calculate?
        # Ideally this should live in self.calculate, but for some 
        # reason super() gets lost if I do that

        start = pd.Timedelta(seconds=trigger-pre)
        stop = pd.Timedelta(seconds=trigger+post)
        psth = signals.loc[:, start:stop]
        psth.columns -= pd.Timedelta(seconds=trigger)

        return psth 


    def _align_time(self, psth):
        if self.agg == 'sum':
            new_psths = psth.T.resample(f"{self.dt}S").sum()
        elif self.agg == 'mean':
            new_psths = psth.T.resample(f"{self.dt}S").mean()
        else:
            # EXPERIMENTAL!
            new_psths = psth.T.resample(f"{self.dt}S").agg(self.agg)

        # TODO: Why doesn't this work
        new_psths = new_psths.interpolate('time', limit=10)
        new_psths = new_psths.reindex(index=self.columns, 
                                      method='nearest') # level=self.name
        new_psths = new_psths.T

        return new_psths


class PSTH(_SingleEventPSTH):
    """Calculates a PSTH for a single experiment based on a list of events or an event_spec. 
    For each neuron, we compute its activity in a pre/post window around event triggers. 

    Initialization parameters
    ----------
    pre : float
        size of pre-window, in seconds
    post : float
        size of post-window, in seconds
    dt : float
        temporal resolution, in seconds. If there is data collected at multiple
        sampling rates, it will all be resampled to dt. 

    Examples
    --------
    >>> # Compute the running PSTH in a peri-3 second window
    >>> strategy = PSTH(pre=3, post=3, dt=0.1)
    >>> signal_spec = {"signal_type": "dfof", "label": "suite2p"}
    >>> event_spec = {"signal_type": "behavior", "event_type": "running", "as_time_events": True}
    >>> psths = strategy.apply_to(
    >>>     expt, signal_spec=signal_spec, events=)
    """
    @save_load
    def apply_to(self, expt : ImagingMixin, events, signal_spec, roi_filter=IndexIdentity, 
                    time_filter=ColumnIdentity, trial_filter=IndexIdentity):
        # signals should just do this
        signals = expt.signals(**signal_spec, as_time_signals=True)
        signals = signals.loc[roi_filter.bind_to(expt)]
        signals.loc[:, ~time_filter.bind_to(expt)] = np.nan
        #signals.columns = expt.imaging_times
        signals.columns = pd.TimedeltaIndex(signals.columns, unit='s')#expt.imaging_times(), unit='s')
        # signals should just do this
        signals.index = signals.index.set_names('roi_label')

        # Instead of passing an event_list directly, can pass an
        # event "spec" to read from experiment
        events = self._load_events(expt, events, trial_filter=trial_filter)

        return self.calculate(signals, events)

    def calculate(self, signals, events, **kwargs):
        # Subclassing makes calling this method very pretty
        return events.apply(super(), signals=signals)

class Responsiveness(PSTH):
    """Calculates mean activity pre and post an event trigger. 
    """
    name = 'Responsiveness'
    def to_columns(self):
        return pd.Index(['pre', 'post'], name=self.name)

    def __init__(self, pre: float = 3., 
                    post: float = 3., dt: float = 0.03):
        super().__init__(pre=pre, post=post, dt=dt)
        self.psth_strategy = PSTH(pre=pre, post=post, dt=dt)

    def calculate(self, signals, events):
        psth = self.psth_strategy.calculate(signals, events)

        if psth is None:
            return None

        trigger = pd.Timedelta(seconds=0)
        pre = psth.loc[:, :trigger].mean(axis=1)
        post = psth.loc[:, trigger:].mean(axis=1)

        return pd.DataFrame({'pre': pre, 'post': post}, columns=self.columns)

    

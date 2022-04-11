import pandas as pd

from lab3.core.filters import Filter, _BoundFilter
from lab3.filters.time_filters import EventFilter

class TrialFilter(Filter):
    """Filter trials (e.g. PSTHs, lap-by-lap tuning curves) on some condition.

    Parameters
    ----------
    trial_name : str
        Trial identifier in the index of the full AnalysisResult dataframe. 
        This will usually be 'OnsetEvent', 'IntervalEvent', 'LapEvent', etc.
    trial_spec : dict
        Keyword arguments passed to expt.events(...) to get the trials
        of interest.
    """
    
    constructor_ids = Filter.constructor_ids + ['trial_name', 'trial_spec']

    def bind(self, trials):
        return _BoundFilter(index=trials, columns=None, 
                            level=self.trial_name,
                            inverse=self.inverse, 
                            name=self.name)

class RunningTrials(TrialFilter):
    def bind_to(self, expt): 
        try:
            as_time_events = self.trial_spec['as_time_events']
        except KeyError:
            as_time_events = False
        
        running_spec = {
            'signal_type': 'behavior',
            'event_type': 'running',
            'fmt': 'interval',
            'as_time_events': as_time_events
        }
        
        running = expt.events(**running_spec)
        
        trials = expt.events(**self.trial_spec)
        
        running_trials = pd.Index([i for i, trial in enumerate(trials) 
                                   if trial.onset in running])
        
        return self.bind(running_trials)

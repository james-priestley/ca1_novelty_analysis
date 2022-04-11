from __future__ import annotations

import inspect
import numpy as np 
import pandas as pd

from abc import abstractmethod
from typing import TYPE_CHECKING

from lab3.core import Analysis
from lab3.core.filters import IndexIdentity, ColumnIdentity

if TYPE_CHECKING:
    from lab3.experiment import ImagingMixin

class BaseSignalAnalysis:

    """Base class for all signal analysis strategies."""

    @classmethod
    def _get_param_names(cls):
        init_signature = inspect.signature(cls.__init__)
        parameters = [p for p in init_signature.parameters.values()
                      if p.name != 'self' and p.kind != p.VAR_KEYWORD]

        return sorted([p.name for p in parameters])

    def get_params(self):
        out = dict()
        for key in self._get_param_names():
            try:
                value = getattr(self, key)
            except AttributeError:
                value = None
            out[key] = value
        return out

    def set_params(self):
        raise NotImplementedError

    def __repr__(self, line_max=80):
        """TODO : Make this pretty"""
        repr_ = type(self).__name__
        repr_ += '('
        base_length = len(repr_)
        curr_line_length = len(repr_)
        for key, value in self.get_params().items():
            param_string = key + '=' + str(value) + ', '
            if (curr_line_length + len(param_string)) <= line_max:
                repr_ += param_string
                curr_line_length += len(param_string)
            else:
                repr_ += '\n' + ' ' * base_length
                repr_ += param_string
                curr_line_length = base_length + len(param_string)
        repr_ = repr_[:-2] + ')'
        return repr_

    def __str__(self):
        return self.__repr__()

class SignalAnalysis(Analysis):
    @abstractmethod
    def apply_to(self, expt : ImagingMixin, signal_spec, roi_filter=None, time_filter=None, **kwargs):
        signals = self._load_signals(expt, signal_spec)
        signals = signals.loc[roi_filter.apply(expt), time_filter.apply(expt)]

        result = self._calculate(signals, **kwargs)
        """
        DO STUFF HERE
        """
        return result

    def _calculate(self, signals, **kwargs):
        """
        DO STUFF HERE
        """
        result.columns = self.columns # suggested, not mandated
        raise NotImplementedError

    def _load_signals(self, expt, signal_spec, roi_filter=IndexIdentity(),
                    time_filter=ColumnIdentity()):
        """Helper method for loading signals.
        """
        signals = expt.signals(**signal_spec)
        signals.index.name = 'roi_label'
        signals = signals.loc[roi_filter.bind_to(expt)]
        signals.loc[:, ~time_filter.bind_to(expt)] = np.nan

        return signals


    def _load_events(self, expt, events, bool_array=False,
                     inverse=False, trial_filter=IndexIdentity):
        """Helper method for loading events.
        """
        try:
            events['signal_type']
            events = expt.events(**events)
            events = events.filter_trials(trial_filter.bind_to(expt))
        except TypeError:
            pass

        if bool_array:
            event_array = events.bool_array(
                            expt.imaging_dataset.num_frames)
            if inverse:
                event_array = ~event_array

            return event_array

        return events


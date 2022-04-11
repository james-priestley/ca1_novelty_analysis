from __future__ import annotations
import pandas as pd
import numpy as np
from functools import reduce
from typing import TYPE_CHECKING

from lab3.core.filters import Filter, TimeFilter, _BoundFilter
from lab3.event import SpatialEvent

from .time_filters import RealTimeEventFilter

class JointFilter(Filter):
    """Implements more complex filtering operations that depend on rows and columns
    e.g. 
    """
    pass

class _BoundJointFilter(_BoundFilter):
    def __call__(self, dataframe):
        return dataframe

class _JointFilterIdentity(Filter):
    def bind_to(self, expt):
        return _BoundJointFilter(index=None, columns=None, level=None)

JointFilterIdentity = _JointFilterIdentity()

class GroupbyFilterMixin(JointFilter):
    def __init__(self, *args, groupby=None, **kwargs):
        self.groupby = groupby
        super().__init__(*args, **kwargs)

    def bind_groupby(self, bound_filters):
        return _BoundGroupbyFilter(bound_filters, groupby=self.groupby)

class ROIEventFilter(GroupbyFilterMixin, RealTimeEventFilter):
    constructor_ids = ['event_spec', 'pre', 'post', 'groupby']

    def bind_to(self, expt):

        events = expt.events(**self.event_spec)
        imaging_times = expt.imaging_times()

        bound_filters = {}

        for roi_events in events:
            idxes = []
            for ev in roi_events:
                lower, = np.where(imaging_times >= ev.onset - self.pre)
                upper, = np.where(imaging_times <= ev.offset + self.post)

                idxes.append(pd.RangeIndex(lower[0], upper[-1]))

            bound_filters[roi_events.name] = self.bind(reduce(pd.Int64Index.union, idxes, pd.Int64Index([])))

        return self.bind_groupby(bound_filters)

class _BoundGroupbyFilter(_BoundJointFilter):

    constructor_ids = []

    def __init__(self, bound_filters, groupby):
        self.bound_filters = bound_filters
        self.groupby = groupby


    def __call__(self, dataframe):
        key = dataframe.level[self.groupby].unique()
        #assert len(key) == 1
        #key = key[0]

        for k in key:
            dataframe.loc[dataframe.level[self.groupby] == k, ~self.bound_filters[k]] = np.nan

        return dataframe


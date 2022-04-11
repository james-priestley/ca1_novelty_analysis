import pandas as pd 
import copy
from abc import abstractmethod
from functools import reduce

from . import Item, Analysis, Group
from lab3.core.helpers import conjunction, disjunction
from lab3.misc import level_accessor

class Filter(Item):
    """Generic filter. Filters a dataframe, usually a signals dataframe (rois or timepoints)
    based on some criterion. Works by implicitly building an AST on the stack.
    
    Usage should be something like
    >>> roi_filter = ROIFilterSubclass(...)
    >>> desired_rois = roi_filter.bind_to(expt)
    >>> time_filter = TimeFilterSubclass(...)
    >>> desired_timepoints = time_filter.bind_to(expt)
    >>> signals.loc[desired_rois, desired_timepoints]
    
    This should be subclassed rather than used directly. 

    IMPORTANT! You MUST use `f1 & f2`, and NEVER `f1 and f2` (likewise for `or`). 
    The `and`/`or` binary operators cannot be overridden in Python, thus 
    `f1 and f2` results in UNDEFINED BEHAVIOR! -Z.

    Read more:
    https://www.python.org/dev/peps/pep-0335/
    """
    inverse = False

    def bind_to(self, item):
        """Defines the indices to select, and binds to an Item, returning a
        _BoundFilter instance
        """
        raise NotImplementedError 

    def __or__(self, other):
        return FilterGroup([self, other], disjunction=True)        
        
    def __add__(self, other):
        return self | other

    def __radd__(self, other):
        if other == 0:
            return self
        else:
            return self | other
    
    def __sub__(self, other):
        return self | -other
    
    def __and__(self, other):
        return FilterGroup([self, other], conjunction=True)
    
    def __mul__(self, other):
        return self & other

    def __xor__(self, other):
        # When would this ever be necessary?
        return (self | other) & ~(self & other)
    
    def __invert__(self):
        newself = copy.deepcopy(self)
        newself.inverse = not self.inverse
        return newself
    
    def __neg__(self):
        return ~self
    
    def __repr__(self):
        if self.inverse:
            return "~" + self.name
        else:
            return self.name

    def _repr_html_(self):
        return repr(self)
    
class _BoundFilter(Filter):
    """Object bound to data and used to slice a DataFrame.
    Created by a Filter, *Not* instantiated directly

    TODO: this^^^ is not an accurate description of what this class is/does
    """
    constructor_ids = ['index', 'columns', 'level', 'inverse', 'name']

    def __call__(self, df):
        if self.index is not None:
            return self.filter_index(df)
        elif self.columns is not None:
            return self.filter_columns(df)

    def filter_index(self, df):
        all_labels = df.index

        # Workaround to pandas bug
        # level= cannot be passed to flat Indexes
        if (self.level is not None) and (len(all_labels.names) > 1):
            # pd.Index.reindex returns some other stuff too - throw it away
            index, _ = all_labels.reindex(self.index, level=self.level)
        else:
            index = self.index

        if self.inverse:
            return all_labels.difference(index)
        else:
            return index

    def filter_columns(self, df):
        all_labels = df.columns
        if self.inverse:
            return all_labels.difference(self.columns)
        else:
            return self.columns

    def __or__(self, other):
        return _BoundFilterGroup([self, other], disjunction=True)        
        
    def __and__(self, other):
        return _BoundFilterGroup([self, other], conjunction=True)

    def __repr__(self):
        return super().__repr__() + "[BOUND]"

class FilterGroup(Filter, Group):
    """Group of filters

    """
    constructor_ids = ['name', 'conjunction', 'disjunction']
    def __init__(self, iterable_of_filters, conjunction=False, 
                    disjunction=False, name='FilterGroup'):
        super().__init__(iterable_of_filters, name=name)
        if not conjunction ^ disjunction:
            raise ValueError("EITHER conjunction or disjunction must be True!")
        self.conjunction = conjunction
        self.disjunction = disjunction
    
    def bind_to(self, *args, **kwargs):
        return _BoundFilterGroup([filt.bind_to(*args, **kwargs) 
                                    for filt in self], 
                                    conjunction=self.conjunction, 
                                    disjunction=self.disjunction)

    def __invert__(self):
        # Distribute using DeMorgan's Laws
        return self.__class__([~filt for filt in self], conjunction=not self.conjunction, 
                                disjunction=not self.disjunction, name=self.name)
        
    def __repr__(self):
        INV = "~"*self.inverse
        
        if self.conjunction:
            return f"{INV}({' ∧ '.join([repr(filt) for filt in self])})"
        elif self.disjunction:
            return f"{INV}({' ∨ '.join([repr(filt) for filt in self])})"

class _BoundFilterGroup(FilterGroup, _BoundFilter):
    def __call__(self, df):
        if self.conjunction:
            return conjunction(self)(df)
        elif self.disjunction:
            return disjunction(self)(df)
        
class ROIFilter(Filter):
    def bind(self, index, level='roi_label'):
        return _BoundFilter(index=index, columns=None, level=level, 
                            inverse=self.inverse, name=self.name)

class TimeFilter(Filter):
    def bind(self, columns):
        return _BoundFilter(index=None, columns=columns, level=None, 
            inverse=self.inverse, name=self.name)

class _IndexIdentity(ROIFilter):
    name = 'IndexIdentity'
    inverse = True
    def bind_to(self, expt):
        return self.bind([], level=None)

    def __call__(self):
        return self

class _ColumnIdentity(TimeFilter):
    name = 'ColumnIdentity'
    inverse = True 
    def bind_to(self, expt):
        return self.bind([])

    def __call__(self):
        return self

IndexIdentity = _IndexIdentity()
ColumnIdentity = _ColumnIdentity()


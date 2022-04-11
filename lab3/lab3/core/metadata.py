import pandas as pd
import numpy as np
from functools import reduce
import warnings

from .classes import Item, Group
from .helpers import wrap

class Metadata(Item):
    """Base class for adding metadata to a dataframe, most likely a signals
    dataframe. 

    Example usecases include: annotating a dendritic imaging dataset with morphological information;
    adding ROI identities determined by post hoc immunohistochemistry to functional datasets; 
    marking cells identified on the red channel; annotating cells as place and non-place cells

    Metadata objects can be layered using the |, &, or + operators (same behavior).

    Examples
    --------
    Add metadata to a signals dataframe:

    >>> meta = SomeMetadata(...)
    >>> meta2 = OtherMetadata(...)
    >>> annotated_signals = expt.signals(signal_type='dfof', label='PYR', metadata = meta | meta2)

    Perform analysis on an annotated dataframe

    >>> signal_spec = {'signal_type': 'dfof', 'label':'PYR', 
    >>>                 'metadata': immuno_data | is_place_cell | is_active}
    >>> result = my_cohort.apply(my_analysis, signal_spec=signal_spec)
    """

    levels = []

    def to_levels(self):
        return self.levels

    def bind(self, expt):
        """Retrieve the metadata from the Experiment object

        Returns
        -------
        pd.Index or pd.MultiIndex
        """
        raise NotImplementedError
    
    def bind_to(self, signals, expt):
        """Retrieve the metadata from the Experiment object, and add it to the signals
        DataFrame.

        Note: This implementation assumes that the output of `bind` is in one-to-one 
        correspondence with the signals DataFrame index. If more sophisticated 
        alignment is needed, this method needs to be overriden. 

        See lab3.metadata.base for examples
        """
        try:
            index = self.bind(expt)
            try:
                return signals.set_index(index, append=True)
            except ValueError:
                # Wrap strategy
                assert len(index) == 1
                dataframe = signals.copy()

                if len(index.names) == 1:
                    return wrap(dataframe, index[0], index.name)
                else:
                    for key, level in zip(index[0], index.names):
                        dataframe = wrap(dataframe, key, header=level)
                    return dataframe

        except Exception as e:
            warnings.warn(f"Unable to add metadata `{str(self)}` because of {e}")
            return signals
    
    def __and__(self, other):
        return MetadataGroup([self, other])
    
    def __add__(self, other):
        return self & other
    
    def __or__(self, other):
        return self & other
    
    def __str__(self):
        return self.name
    
    def _repr_html_(self):
        return pd.DataFrame([], index=pd.MultiIndex.from_frame(
            pd.DataFrame([], columns=self.to_levels())))._repr_html_()

class MetadataGroup(Metadata, Group):
    def to_levels(self):
        return sum([m.to_levels() for m in self], [])

    def bind(self, expt):
        return _deduplicate(pd.MultiIndex.from_frame(
            _join_indexes([m.bind(expt) for m in self])))
    
    def bind_to(self, signals, expt):        
        return _deduplicate_index(reduce(lambda x, y: y.bind_to(
            x.bind_to(signals, expt), expt), self))
    
    def __str__(self):
        return " | ".join([str(m) for m in self])
    
class _DefaultMetadata(Metadata):
    def bind(self, expt):
        return pd.Index([])
    
    def bind_to(self, signals, expt):
        return signals

    def __call__(self, *args, **kwargs):
        return self

DefaultMetadata = _DefaultMetadata()

def _join_indexes(indexes):
    return reduce(lambda x,y: x.join(y.to_frame(index=False)), indexes, 
           pd.DataFrame([[]]))

def _deduplicate(index):
    index_names = pd.Index(index.names)
    where_duplicated = np.where(index_names.duplicated())[0]
    return index.droplevel(level=list(where_duplicated))

def _deduplicate_index(dataframe):
    df = dataframe.copy()
    index = dataframe.index
    dedup = _deduplicate(index)
    df.index = dedup
    return df

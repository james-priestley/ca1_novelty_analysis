import abc
import pickle as pkl 
import pandas as pd
import numpy as np

from sklearn.preprocessing import scale

from lab3.core.filters import ROIFilter, ColumnIdentity
from lab3.analysis.spatial.abstract_place_fields import PlaceFields
from lab3.analysis.basic.psth import Responsiveness

from lab3.experiment.base import ImagingExperiment

class ClassifierROIFilter(ROIFilter):
    
    constructor_ids = ['feature_extractor', 'trained_classifier', 'signal_spec']
    
    @classmethod
    def load_pkl(cls, path_to_pkl, signal_spec):
        with open(path_to_pkl, "rb") as f:
            pickled_filter = pkl.load(f)
        
        return cls(feature_extractor=pickled_filter['feature_extractor'], 
                   trained_classifier=pickled_filter['trained_classifier'], 
                   signal_spec=signal_spec)
    
    def bind_to(self, expt):
        X = self.feature_extractor.apply_to(expt, signal_spec=self.signal_spec)
        X = X.fillna(0)
        X = pd.DataFrame(scale(X), index=X.index, columns=X.columns)
        y = pd.Series(self.trained_classifier.predict(X), index=X.index)
        
        return self.bind(y[y>0].index)

class ResponsivenessFilter(ROIFilter, Responsiveness):
    constructor_ids = Responsiveness.constructor_ids + ['threshold', 'signal_spec', 'event_spec']
    
    def __init__(self, signal_spec, event_spec, threshold=0.5, **kwargs):
        self.signal_spec = signal_spec
        self.event_spec = event_spec
        self.threshold = threshold
        
        super().__init__(**kwargs)

    @property
    def name(self):
        return f"Is{self.event_spec['event_type'].capitalize()}Responsive"

        
    def _repr_html_(self):
        return ROIFilter.__repr__(self)
    
    def bind_to(self, expt : ImagingExperiment):
        responses = super().apply_to(expt, signal_spec=self.signal_spec, 
                                     events=self.event_spec)
        responses = responses.groupby('roi_label').agg('mean')
        scalar_responses = responses['post'] - responses['pre']
        is_responsive = scalar_responses[scalar_responses > self.threshold].index
        return self.bind(is_responsive)

class IsPlaceCell(ROIFilter, PlaceFields):
    constructor_ids = PlaceFields.constructor_ids + ['signal_spec', 'time_filter']

    def __init__(self, *args, signal_spec={}, time_filter=ColumnIdentity, **kwargs):
        self.signal_spec = signal_spec
        self.time_filter = time_filter
        super().__init__(*args, **kwargs)

    def bind_to(self, expt):
        pfs = super().apply_to(expt, signal_spec=self.signal_spec, 
                                time_filter=self.time_filter)
        return self.bind(index=pfs.level['roi_label'].unique())

class IsInterneuron(ClassifierROIFilter):
    def __init__(self, signal_spec={"signal_type": "dfof"}):
        with open('/data/Zhenrui/classifiers/is_interneuron.pkl', 'rb') as f:
            pickled_filter = pkl.load(f)
        
        super().__init__(feature_extractor=pickled_filter['feature_extractor'], 
                   trained_classifier=pickled_filter['trained_classifier'], 
                   signal_spec=signal_spec)

class PlaceFieldLocationFilter(IsPlaceCell):
    """Filter cells by the location of their place fields
    
    Examples
    --------
    Select cells with place fields within the zone of interest.
    >>> place_field_starts = PlaceFieldStart(signal_spec=signal_spec)
    >>> place_field_stops = PlaceFieldStop(signal_spec=signal_spec)
    >>> zone_of_interest = (0.2, 0.4)
    >>> pcs_in_zone = (0.2 < place_field_starts) & (place_field_stops < 0.4)
    >>> 
    
    Important: Remember the belt is circular! Obviously there does not exist 
    a well-ordering on the circle. It is up to you to handle behavior at the reset.
    """
    constructor_ids = IsPlaceCell.constructor_ids + ['lower', 'upper']
    def __init__(self, *args, lower=0, upper=1, **kwargs):
        self.lower = lower
        self.upper = upper
        super().__init__(*args, **kwargs)
    
    @property
    def name(self):
        if self.lower == 0:
            return f"({self.__class__.__name__} < {self.upper})"
        elif self.upper == 1:
            return f"({self.lower} < {self.__class__.__name__})"
        else:
            return f"({self.lower} < {self.__class__.__name__} < {self.upper})"
        
    def __gt__(self, position):
        cons = self.constructors()
        cons['lower'] = position
        return self.__class__(**cons)
    
    def __lt__(self, position):
        cons = self.constructors()
        cons['upper'] = position
        return self.__class__(**cons)
    
    def __ge__(self, position):
        return ~(self < position)
    
    def __le__(self, position):
        return ~(self > position)
    
    def _repr_html_(self):
        raise NotImplementedError
    
    def pos2bin(self, position):
        return np.searchsorted(np.linspace(0,1,self.num_bins), position, side='right') - 1
    
    @abc.abstractmethod
    def bind_to(self, expt):
        """"""

class PlaceFieldStart(PlaceFieldLocationFilter):
    def bind_to(self, expt):
        pfs = super().apply_to(expt, signal_spec=self.signal_spec)
        
        lower_bin = self.pos2bin(self.lower)
        upper_bin = self.pos2bin(self.upper)
        
        pfs = pfs.loc[(lower_bin < pfs.field_start) & (pfs.field_start < upper_bin)]
        
        return self.bind(index=pfs.level['roi_label'].unique())
    
class PlaceFieldStop(PlaceFieldLocationFilter):
    def bind_to(self, expt):
        pfs = super().apply_to(expt, signal_spec=self.signal_spec)
        
        lower_bin = self.pos2bin(self.lower)
        upper_bin = self.pos2bin(self.upper)
        
        pfs = pfs.loc[(lower_bin < pfs.field_stop) & (pfs.field_stop < upper_bin)]
        
        return self.bind(index=pfs.level['roi_label'].unique())

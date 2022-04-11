import numpy as np 
import scipy.stats as stats 
import statsmodels
from statsmodels.tsa.statespace import sarimax
from statsmodels.regression.linear_model import yule_walker

from abc import abstractmethod
import pandas as pd 

import lab3

from lab3.core.filters import IndexIdentity, ColumnIdentity

from lab3.analysis.base import SignalAnalysis
from lab3.analysis.basic.psth import Responsiveness
from lab3.analysis.spatial.abstract_spatial_tuning import SpatialTuning

from lab3.core.helpers import save_load

class Feature(type):
    def __new__(cls, func, name=None):
        if name is None:
            name = func.__name__.capitalize()
        
        return type(name, (FeatureExtractor,), 
                    {
                        'func': staticmethod(func), 
                     '__doc__': func.__doc__,
                     '__module__': __name__ #lab #lab3.analysis.basic.features
                    }
                   )
    
class SpatialFeature(type):
    def __new__(cls, func, name=None):
        if name is None:
            name = func.__name__.capitalize()
        
        return type(name, (SpatialFeatureExtractor,), 
                    {
                        'func': staticmethod(func), 
                     '__doc__': func.__doc__,
                     '__module__':__name__ #lab# lab3.analysis.basic.features
                    }
                   )

class ResponsivenessFeature(type):
    def __new__(cls, event_spec, name=None):
        if name is None:
            name = event_spec['event_type'].capitalize()
        
        return type(name, (ResponsivenessFeatureExtractor,), 
                    {
                        'events': event_spec, 
                        '__module__': __name__ #lab# lab3.analysis.basic.features
                    }
                   )
    
class FeatureExtractor(SignalAnalysis):
    """Generic class for feature extraction, to feed to machine learning
    """
    constructor_ids = ['roi_filter', 'time_filter']
    def to_columns(self):
        return pd.Index([self.name])
        
    def __init__(self, roi_filter=IndexIdentity(), time_filter=ColumnIdentity()):
        self.name = self.__class__.__name__
        self.roi_filter = roi_filter
        self.time_filter = time_filter
        
    @save_load
    def apply_to(self, expt, signal_spec):
        signals = self._load_signals(expt, signal_spec)
        signals = signals.loc[self.roi_filter.bind_to(expt), 
                              self.time_filter.bind_to(expt)]
        
        return self.calculate(signals)
    
    def calculate(self, signals):
        result = signals.apply(self.func, axis=1)
        try:
            result = result.to_frame()
        except AttributeError:
            pass
        result.columns = self.columns
        return result
    
    
class SpatialFeatureExtractor(SpatialTuning, FeatureExtractor):
    
    constructor_ids = SpatialTuning.constructor_ids + FeatureExtractor.constructor_ids
    
    def to_columns(self):
        return pd.Index([self.name])
    
    def __init__(self, nbins: int = 100, sigma: float = 3, 
                 normalize: bool = True, roi_filter=IndexIdentity(), 
                 time_filter=ColumnIdentity()):
        self.name = self.__class__.__name__
        self.roi_filter = roi_filter
        self.time_filter = time_filter
        self.nbins = nbins 
        self.bins = None
        self.sigma = sigma 
        self.normalize = normalize
        self.roi_filter = roi_filter
        self.time_filter = time_filter
        self.spatial_tuning_strategy = SpatialTuning(nbins=nbins, bins=None, sigma=sigma, 
                                                        normalize=normalize)
        
    def apply_to(self, expt, signal_spec):
        tuning_curves = self.spatial_tuning_strategy.apply_to(
                                        expt, signal_spec=signal_spec, 
                                         roi_filter=self.roi_filter, 
                                         time_filter=self.time_filter)
        tuning_curves = tuning_curves.groupby('roi_label').apply(np.mean)
        tuning_curves = tuning_curves.fillna(0) # Set nans to 0
        return self.calculate(tuning_curves)
    
    def calculate(self, tuning_curves):
        result = tuning_curves.apply(self.func, axis=1).to_frame()
        result.columns = self.columns
        return result

class ResponsivenessFeatureExtractor(FeatureExtractor, Responsiveness):
    constructor_ids = Responsiveness.constructor_ids + FeatureExtractor.constructor_ids + ['contrast']
    
    # def to_columns(self):
    #   return pd.Index([self.name])

    def __init__(self, pre: float = 3., post: float = 3., dt: float = 0.03, contrast: bool = False,
                 roi_filter=IndexIdentity(), time_filter=ColumnIdentity()):
        self.name = self.__class__.__name__
        self.response_strategy = Responsiveness(pre=pre, post=post, dt=dt)
        self.pre = pre 
        self.post = post 
        self.dt = dt 
        self.roi_filter = roi_filter
        self.time_filter = time_filter
        self.contrast = contrast
    
    def apply_to(self, expt, signal_spec):
        pre_post = self.response_strategy.apply_to(expt, events=self.events, signal_spec=signal_spec, 
                                    roi_filter=self.roi_filter, time_filter=self.time_filter)

        groupby = [label for label in pre_post.index.names if 'Event' not in label]
        pre_post = pre_post.groupby(groupby).mean()

        responsiveness = pre_post['post'] - pre_post['pre']

        if self.contrast:
            responsiveness /= pre_post['post'] + pre_post['pre']

        responsiveness = responsiveness.to_frame()
        responsiveness.columns = self.columns

        return responsiveness

class Moment(FeatureExtractor):
    constructor_ids = ['moment', 'roi_filter', 'time_filter']
    
    def to_columns(self):
        return pd.Index([f"Moment[{self.moment}]"])
    
    def __init__(self, moment, roi_filter=IndexIdentity(), time_filter=ColumnIdentity()):
        self.moment = moment
        super().__init__(roi_filter=roi_filter, time_filter=time_filter)
        
    def func(self, x):
        return stats.moment(x, moment=self.moment, nan_policy='omit')
        
class Cumulant(FeatureExtractor):
    constructor_ids = ['cumulant', 'roi_filter', 'time_filter']
    
    def to_columns(self):
        return pd.Index([f"Cumulant[{self.cumulant}]"])
    
    def __init__(self, cumulant, roi_filter=IndexIdentity(), time_filter=ColumnIdentity()):
        self.cumulant = cumulant
        assert 1<= cumulant <= 4,\
            "Only cumulants up to 4 are implemented!"
        super().__init__(roi_filter=roi_filter, time_filter=time_filter)
        
    def func(self, x):
        return stats.kstat(x, n=self.cumulant)

class ARCoef(FeatureExtractor):
    constructor_ids = ['p', 'method', 'roi_filter', 'time_filter']
    
    def to_columns(self):
        try:
            p = self.p[0]
            d = self.p[1]
        except TypeError:
            p = self.p
            d = 0
        
        p_features = [f'AR[{i}]' for i in range(p)]
        d_features = [f'MA[{i}]' for i in range(d)]    
        
        return pd.Index(p_features + d_features)
    
    def __init__(self, p, method='yule_walker', roi_filter=IndexIdentity(), 
                 time_filter=ColumnIdentity(), **ar_kwargs):
        """Fits an AR(p) model to the signals and extracts the AR coefficients
        as features.
        
        Inputs
        -------
        p : int or tuple of int
            The order of the autoregressive process
        method : str, 'yule_walker' or 'sarimax'
            If you don't know what these are, use 'yule_walker'
        roi_filter : ROIFilter
        time_filter : TimeFilter
        **ar_kwargs : dict
            Keyword arguments passed to either yule_walker or SARIMAX 
        """
        self.p = p
        self.ar_kwargs = ar_kwargs
        self.method = method
        
        if self.method == 'yule_walker':
            self.func = self.yule_walker
        elif self.method == 'sarimax':
            self.func = self.sarimax
        
        super().__init__(roi_filter=roi_filter, time_filter=time_filter)
    
    def yule_walker(self, x):
        return pd.Series(yule_walker(x, order=self.p, **self.ar_kwargs)[0])
    
    def sarimax(self, x):
        mod = sarimax.SARIMAX(x, order=self.p, **self.ar_kwargs)
        res = mod.fit()
        return np.concatenate([res.arparams, res.maparams])
    
# NB: It's important here that the class name and the `name` attribute MATCH!
# Otherwise pickle-dependent operations (including parallelization) will break

Mean = Feature(np.nanmean, name="Mean")
StandardDeviation = Feature(np.nanstd, name="StandardDeviation")
Variance = Feature(np.nanvar, name="Variance")
Max = Feature(np.nanmax, name="Max")
Min = Feature(np.nanmin, name="Min")
Median = Feature(np.nanmedian, name="Median")
#Mode = Feature(stats.mode)
Skew = Feature(stats.skew)
Kurtosis = Feature(stats.kurtosis)
IQR = Feature(stats.iqr)
GMean = Feature(stats.gmean)
HMean = Feature(stats.hmean)

Entropy = SpatialFeature(stats.entropy)
Peak = SpatialFeature(np.argmax, name="peak")
MAD = SpatialFeature(stats.median_absolute_deviation, name="MAD")

RunningStart = ResponsivenessFeature({'signal_type': 'behavior', 
                                    'event_type': 'running', 
                                    'fmt': 'onset'}, name='RunningStart')
RunningStop = ResponsivenessFeature({'signal_type': 'behavior', 
                                    'event_type': 'running', 
                                    'fmt': 'onset'}, name='RunningStop')
Reward = ResponsivenessFeature({'signal_type': 'behavior', 
                                'event_type': 'reward', 
                                'fmt': 'onset'})
Water = ResponsivenessFeature({'signal_type': 'behavior', 
                                'event_type': 'water', 
                                'fmt': 'onset'})
LED_on = ResponsivenessFeature({'signal_type': 'behavior', 
                                'event_type': 'led_context_pin', 
                                'fmt': 'onset'}, name='LED_on')
LED_off = ResponsivenessFeature({'signal_type': 'behavior', 
                                'event_type': 'led_context_pin', 
                                'fmt': 'offset'}, name='LED_off')
Licking = ResponsivenessFeature({'signal_type': 'behavior', 
                                    'event_type': 'licking', 
                                    'fmt': 'onset'})

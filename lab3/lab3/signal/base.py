"""Base signal classes"""

import inspect
from pandas import Series, DataFrame, HDFStore


class SignalFile(HDFStore):

    pass
    # def __init__(self, *args, **kwargs):
    #     super().__init__(self, *args, **kwargs)


class SignalFrame(DataFrame):

    """doc string"""

    def __init__(self, signals, metadata={}, **kwargs):
        super().__init__(signals)
        self.metadata = metadata
    #     self.cls = type(signals[0])
    #
    # def indexed_signal(self, index):
    #     return self.cls(self.iloc[index], **self.metadata.iloc[index])


class CalciumSignals(SignalFrame):
    pass


class DFOFSignals(SignalFrame):
    pass


class LFPSignals(SignalFrame):
    pass


class SpikeSignals(SignalFrame):
    pass


class Signal(Series):
    pass


class BaseSignalTransformer:

    """Base class for all signal transformation strategies. Adapted largely
    from sklearn.base.BaseEstimator"""

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

    def __get_state__(self):
        pass

    def __set_state__(self):
        pass

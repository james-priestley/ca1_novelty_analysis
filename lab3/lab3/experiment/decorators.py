"""Convenience function wrappers for experiment methods"""

import inspect
from functools import wraps

from sima.misc import most_recent_key


def get_default_args(func):
    signature = inspect.signature(func)
    return {
        k: v.default
        for k, v in signature.parameters.items()
        if v.default is not inspect.Parameter.empty
    }


def default_label_handling(func):
    """Decorator for selecting a default ROI label if None is passed. Defaults
    to the roi label for the most recently created ROIList in the imaging
    dataset

    The wrapped function should take an ImagingMixin-derived object as an
    argument and a label as a keyword argument.
    """

    @wraps(func)
    def set_default_label(*args, **kwds):

        kwargs = get_default_args(func)
        kwargs.update(kwds)

        if kwargs['label'] is None:
            kwargs['label'] = most_recent_key(args[0].imaging_dataset.ROIs)
            print(f'Label not passed. Using label {kwargs["label"]}')

        return func(*args, **kwargs)

    return set_default_label

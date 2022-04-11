import pandas as pd
import pprint
import os
import json
import pickle as pkl
from functools import reduce
import inspect
import warnings

# O(|MRO_1|+|MRO_2|+|MRO_3|+...)
def infer_item_class(iterable_of_items, typecheck=False):
    if len(iterable_of_items) == 0:
        return None

    classes = [item.__class__.mro() for item in iterable_of_items]

    if typecheck and len(set(map(tuple, classes))) != 1:
        raise TypeError('All items in group must be of same type!')

    return infer_lowest_common_ancestor(classes)


def infer_lowest_common_ancestor(list_of_classes):
    if len(list_of_classes) == 1:
        return list_of_classes[0][0]
    else:
        shared = common_ancestors(list_of_classes[0], list_of_classes[1])
        return infer_lowest_common_ancestor([shared] + list_of_classes[2:])


def common_ancestors(ancestors1, ancestors2):
    ancestor_set = set(ancestors2)
    shared = []
    for a in ancestors1:
        if a in ancestor_set:
            shared.append(a)
    return shared

def wrap(df, key, header=None):
    return pd.concat([df], keys=[key], names=[header])

def bind_args(cls, args, kwargs):
    return cls(*args, **kwargs)

def check_instance(obj, cls):
    try:
        return isinstance(obj, cls)
    except TypeError:
        from lab3.core.registry import lookup_class
        cls = lookup_class(cls)
        return isinstance(obj, cls)

def is_target_object(arg, function, name=None):
    """Checks if `arg` is the target object of `function`
    based on static type hints in the function signature
    Examples
    --------
    >>> def foo(a: int):
    >>>     return a
    >>> is_target_object(foo, 4) # returns True
    >>> is_target_object(foo, 'attila') # returns False

    If the function is not type-annotated, returns `None`
    with a warning:
    >>> def bar(b):
    >>>     return b
    >>> is_target_object(bar, 4) # returns None

    Returns
    --------
    {True, False} or None if untyped
    """
    signature = inspect.signature(function)
    params = signature.parameters
    keys = list(params.keys())

    if keys[0] != 'self':
        first_arg = params[keys[0]]
    else:
        first_arg = params[keys[1]]

    if first_arg.annotation is inspect._empty:
        if name is None:
            name = function.__name__
        warnings.warn(f"The Analysis `{name}` is not type-annotated "
                      "(consider adding annotations to allow type-checking)")
        return None
    elif check_instance(arg, first_arg.annotation):
        return True
    else:
        return False

def conjunction(list_of_filters):
    return lambda df: reduce(lambda x,y: x & y, 
                        [filt(df) for filt in list_of_filters])

def disjunction(list_of_filters):
    return lambda df: reduce(lambda x,y: x | y, 
                        [filt(df) for filt in list_of_filters])

def _params2str(raw_params):
    params = {}
    for key, value in raw_params.items():
        if isinstance(value, dict):
            params[key] = _params2str(value)
        else:
            params[key] = str(value)
    return params

def stringify_params(raw_params):
    return json.dumps(_params2str(raw_params), sort_keys=True)

class ResultStore(pd.HDFStore):
    def __init__(self, path, hash_size=16, **kwargs):
        self.hash_size = hash_size
        self.h5_path = os.path.join(path, 'analysis_results.h5')
        self.pkl_path = os.path.join(path, 'result_parameters.pkl')
        
        try:
            with open(self.pkl_path, 'rb') as f:
                self._pkl_data = pkl.load(f)
        except (EOFError, FileNotFoundError, IOError):
            self._pkl_data = {}        
        
        super().__init__(self.h5_path, mode='a', **kwargs)
        
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        
    def close(self):
        try:
            self._pkl_handle.close()
        except AttributeError:
            pass
            
        super().close()
        
    def get_hash(self, parameters):
        return str(hex(hash(stringify_params(parameters)) % 2**self.hash_size))
    
    def _update_pkl(self, data):
        with open(self.pkl_path, mode='wb+') as f:
            pkl.dump(data, f)
    
    def info(self):
        info = super().info()
        
        legend = stringify_params(self._pkl_data)
        
        return info + "\nLEGEND\n" + pprint.pformat(legend)
        
    def put(self, parameters, dataframe):
        hash_val = self.get_hash(parameters)            
        self._pkl_data[hash_val] = parameters
        self._update_pkl(self._pkl_data)
        
        super().put(hash_val, pd.DataFrame(dataframe))
    
    def get(self, parameters):
        hash_val = self.get_hash(parameters)
        return super().get(hash_val)
    
def save_load(apply_to):
    def save_loadable_apply_to(self, expt, verbose=False, **runtime_parameters):
        try:
            if self.load_saved:
                return self.load(expt, verbose=self.verbose, **runtime_parameters)
        except Exception as e:
            pass
            
        result = apply_to(self, expt, **runtime_parameters)
        
        try:
            if self.save_results and not self.loaded:
                self.save(result, expt, verbose=self.verbose, **runtime_parameters)
        except Exception as e:
            pass        
        return result
    
    return save_loadable_apply_to
    

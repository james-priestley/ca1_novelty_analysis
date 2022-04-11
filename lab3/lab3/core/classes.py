"""Core set of classes that form the basis of most objects in the LAB3 module.
"""

from abc import ABCMeta, abstractmethod

import numpy as np
import pandas as pd
import itertools as it
from pandas.core.dtypes.common import is_scalar

from lab3.misc import level_accessor
from lab3.core.helpers import infer_item_class, bind_args, is_target_object, ResultStore

class ConstructorMeta(ABCMeta):

    """Custom metaclass to force the user to set all constructor ids."""

    constructor_ids = []

    def __call__(self, *args, **kwargs):
        obj = super().__call__(*args, **kwargs)

        if not hasattr(obj, 'name'):
            obj.name = obj.__class__.__name__

        for attr_name in obj.constructor_ids:
            try:
                getattr(obj, attr_name)
            except AttributeError:
                raise AttributeError(
                    f'Required constructor attribute ({attr_name}) not set')
        return obj


class Item(metaclass=ConstructorMeta):

    """Abstract base class for modeling single entities. When implementing a
    subclass of Item, you must determine a list of attributes shared by all
    instances of the class, which can be used to uniquely reconstruct each
    item instance. This list is defined as the class attribute
    `constructor_ids`.

    Subclasses of Item cannot be instantiated unless all attributes specified
    in `constructor_ids` are set (i.e. via __init__).

    Examples
    --------
    Create a new Item class:

    >>> class Person(Item):
    >>>     constructor_ids = ['name', 'position']
    >>>     def __init__(self, name, position='grad_student'):
    >>>         self.name = name
    >>>         self.position = position

    Instantiate the item and run an analysis on the object instance

    >>> obj = Person('zhenrui', position='grad_student')
    >>> obj.apply(...)  # pass an instance of an analysis class here
    """

    # List of ids sufficient to reconstruct the item uniquely
    constructor_ids = []

    def __init__(self, **constructors):
        """This will set all keyword arguments as instance attributes. In
        practice, you should implement `__init__` in derived classes"""
        self._register_kwargs(constructors)

    def _register_kwargs(self, kwargs):
        self.__dict__.update(kwargs)

    def constructors(self):
        """Returns a dict of parameters sufficient to uniquely identify
        and instantiate this object
        """
        return {
            cid: self.__dict__[cid] for cid in self.constructor_ids
        }

    def label(self):
        """Returns scalar identifier"""
        # Why should we assume this is just the first constructor value?
        try:
            return list(self.constructors().values())[0]
        except Exception:  # what error should we be catching here?
            return None

    def identifiers(self):
        """Return a dict of identifiers (for information display only)"""
        return self.constructors()

    def apply(self, analysis, *args, groupby=None, agg=None, **kwargs):
        """doc"""
        result = analysis.apply_to(self, *args, **kwargs)

        if groupby is not None and agg is not None:
            return result.groupby(groupby).agg(agg)
        else:
            return result

    def is_target_object(self, of):
        return is_target_object(self, function=of.apply_to, name=of.name)

    @property
    def _df(self):
        return pd.DataFrame(np.zeros((1, 0)))

    def __hash__(self):
        return hash(frozenset(self.constructors().items()))

    def __eq__(self, other):
        return hash(self) == hash(other)

    # def __repr__(self):
    #     return "<{} {}>".format(self.__class__.__name__, ' '.join(
    #         [f'{key}="{value}"' for key, value in self.identifiers().items()]))

    def __repr__(self, line_max=80):
        """TODO : Make this pretty"""
        repr_ = type(self).__name__
        repr_ += '('
        base_length = len(repr_)
        curr_line_length = len(repr_)
        for key, value in self.identifiers().items():
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
        return repr(self)

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, d):
        self.__dict__.update(d)


class Group(list, Item):

    """Class for modeling a collection of Items of homogenous type. A Group is
    also an Item, and so Groups can be collections of Groups, allowing for
    heirarchical organization of objects.

    Examples
    --------
    Assume we have defined an Item subclass `Person`.
    Nesting collections of Items:

    >>> people = [Person('zhenrui'), Person('james'), Person('jack')]
    >>> grp = Group(iterable_of_items=people, name='code_review')
    >>> grp_of_grps = Group([grp] * 2)

    Indexing is straightforward. This returns a group:

    >>> grp_of_grps[0]

    This returns a Person:

    >>> grp_of_grps[0][0]

    Slicing returns a group:

    >>> grp_of_grps[0][0:2]
    >>> grp_of_grps[0:2][0:2]

    Since Groups are Items, we may define analysis classes that operate on
    Groups, and apply these with the same syntax as regular Items:

    >>> grp.apply(...)
    >>> grp_of_grps.apply(...)

    By subclassing Group, you can augment it with functionality specific
    to certain item classes:

    >>> class Meeting(Group):
    >>>     constructor_ids = ['name', 'meeting_type']
    >>>     def __init__(self, people, meeting_type='general', **kwargs):
    >>>         assert all([isinstance(p, Person) for p in people]), \
    >>>             'All elements of "people" must be Person objects!'
    >>>         super().__init__(iterable_of_items=people, **kwargs)
    >>>         self.meeting_type = meeting_type
    >>>
    >>>     def my_custom_method(self):
    >>>         # Here I implement a function specific to collections of
    >>>         # Person objects
    """

    constructor_ids = ['name', 'parallelize']

    def __init__(self, iterable_of_items=(), parallelize=False,
                 item_class=None, name='unnamed', **kwargs):
        super().__init__(iterable_of_items)
        if item_class is None:
            # should this be a class attribute or abstract property?
            self.item_class = infer_item_class(iterable_of_items)
        else:
            self.item_class = item_class
        self.parallelize = parallelize
        self.name = name

        self._register_kwargs(kwargs)

    @property
    def _df(self):
        return pd.concat([self._box(item) for item in self])

    def apply(self, analysis, *args, groupby=None, agg=None, **kwargs):
        """By default no groupby"""
        return AnalysisResult(self, analysis, *args, groupby=groupby, agg=agg, **kwargs)
    @property
    def index(self):
        return self._df.index

    @classmethod
    def from_index(cls, index, name=None):
        frame = index.to_frame(index=False)
        return cls.from_frame(frame, name=name)

    @classmethod
    def from_frame(cls, frame, name=None):
        # This MUST stay here or a circular import will result!
        from lab3.core.registry import lookup_class

        level_name = frame.columns[0]
        if len(frame.columns) > 1:
            return Group([Group.from_frame(df.drop(level_name, axis=1),
                                           name=group_name)
                          for group_name, df in frame.groupby(level_name)],
                         name=name, item_class=lookup_class(level_name))
        else:
            return Group.from_scalars(frame[level_name], name=name,
                                      item_class=lookup_class(level_name))

    def level(self, *args, **kwargs):
        return self.from_frame(self._df.level(*args, **kwargs).reset_index(),
                               name=self.name)

    @classmethod
    def from_scalars(cls, scalars, item_class, name=None, **kwargs):
        # This only works if item_class only has one position arg to be set!
        return cls([item_class(x) for x in scalars], name=name, **kwargs)

    def _box(self, item):
        boxed_df = pd.concat([item._df], keys=[item.label()],
                             names=[self.item_class.__name__])
        if None in boxed_df.index.names:
            boxed_df.index = boxed_df.index.droplevel(None)

        return boxed_df

    def __add__(self, other):
        return self.__class__(self + other)

    def __getitem__(self, key):
        if is_scalar(key):
            return super().__getitem__(key)
        else:
            return self.__class__(super().__getitem__(key))

    def __repr__(self):
        return self.index.to_frame(index=None).__repr__()

    def _repr_html_(self):
        return f"<div><b>{self.__class__.__name__}: " \
               + f"'{self.label()}'</b></div>" + self._df._repr_html_()


class Analysis(Item):

    """Abstract base class for designing analyses."""

    @abstractmethod
    def to_columns(self):
        """Define a priori what columns are expected in the output.
        This should NOT take any parameters but only use what is available
        as an instance attribute.

        Returns
        -------
        pd.Index
        """

    @abstractmethod
    def apply_to(self, item, *args, **kwargs):
        """Specifies how to go from an Item (Event, Experiment, Mouse, Group,
        etc.) to results.

        Should load the data from item, preprocess *args and **kwargs,
        and call self.calculate

        TODO: Maybe this can be formalized into abstract methods
        """

    @abstractmethod
    def calculate(self, data, *args, **kwargs):
        """Implement the analysis algorithm here. Unlike `apply`, this should
        take a dataframe and return a dataframe.
        """

    @property
    def columns(self):
        return self.to_columns()

    @property
    def _df(self):
        return pd.DataFrame([], columns=self.columns)

    def get_params(self):
        params = {}
        for key, val in self.__dict__.items():
            if key[0] != '_':
                params[key] = val
        return params

    # def __repr__(self, line_max=80):
    #     """TODO : Make this pretty"""
    #     repr_ = type(self).__name__
    #     repr_ += '('
    #     base_length = len(repr_)
    #     curr_line_length = len(repr_)
    #     for key, value in self.get_params().items():
    #         param_string = key + '=' + str(value) + ', '
    #         if (curr_line_length + len(param_string)) <= line_max:
    #             repr_ += param_string
    #             curr_line_length += len(param_string)
    #         else:
    #             repr_ += '\n' + ' ' * base_length
    #             repr_ += param_string
    #             curr_line_length = base_length + len(param_string)
    #     repr_ = repr_[:-2] + ')'
    #     return repr_

    def _repr_html_(self):
        return pd.DataFrame([], columns=self.columns)._repr_html_()


    def save(self, df, expt, verbose=False, **runtime_parameters):        
        parameters = {"analysis": self, "runtime_parameters": runtime_parameters}
        
        with ResultStore(expt.sima_path) as f:
            f.put(parameters, df)
            
        message = f'Saving results of analysis: {type(self).__name__} \n on experiment {expt}'
        if self.verbose:
            print(message)
    
    def load(self, expt, verbose=False, **runtime_parameters):        
        parameters = {"analysis": self, "runtime_parameters": runtime_parameters}
        
        with ResultStore(expt.sima_path) as f:
            data = f.get(parameters)
            
        self.loaded = True
        
        message = f'Loaded results of analysis: {type(self).__name__} \n on experiment {expt}'
        if self.verbose:
            print(message)
            
        return data

class Automorphism(Analysis):

    """Abstract parent class for analyses which are automorphisms, i.e.,
    they take a dataframe and return a dataframe of the same shape.

    As a result, columns may be unknown a priori. A consequence of this is that
    these analyses cannot necessarily be joined when mapped across a Group
    """
    def to_columns(self):
        return pd.Index(["__IDEM__"])

class AnalysisGroup(Analysis, Group): 
    """Join multiple analyses together. Allows easy creation of matrices.
    """
    def to_columns(self):
        cols = [pd.concat([analysis._df], keys=[analysis.name], 
                        names=['Analysis', 'Feature'], axis=1)
                    for analysis in self]
        
        return pd.concat(cols, axis=1).columns
    
    def apply_to(self, item, *args, args_list=(), kwargs_list=(), 
              **kwargs):
        try:
            args_list[0][0]
        except IndexError:
            args_list = it.repeat(args)
        try:
            kwargs_list[0]
        except IndexError:
            kwargs_list = it.repeat(kwargs)
        
        results = [analysis.apply_to(item, *args, **kwargs) 
                       for analysis, args, kwargs in zip(self, args_list, kwargs_list)]
        try:
            result = pd.concat(results, axis=1)
            result.columns = self.columns
            return result
        except:
            return results

class AnalysisResult(pd.DataFrame):

    _metadata = ['group', 'analysis', 'error_policy', '_groupby', '_agg']

    def __init__(self, group=None, analysis=None, data=None, *args, groupby=None, agg=None, 
                 error_policy='warn', **kwargs):

        if analysis is not None:
            self.group = group 
            self.analysis = analysis 
            self._groupby = groupby
            self._agg = agg
            self.error_policy = error_policy
            
            # TODO: Maybe this can happen lazily/later
            # TODO: Maybe args and kwargs shouldn't get passed here
    #        super().__init__(index=group, columns=analysis)

            data = self.calculate(*args, **kwargs)
            super().__init__(data)
        elif data is not None:
            super().__init__(data=data, **kwargs)
        else:
            # Fall-through case
            super().__init__(group, **kwargs)
        
    @property
    def _constructor(self):
        return AnalysisResult

    def calculate(self, *args, **kwargs):
        """By default no groupby
        """
        if self.group.parallelize is True:
            map_df = self._map_parallel(self.analysis, *args, **kwargs)
        else:
            map_df = self._map(self.analysis, *args, **kwargs)

        if self._groupby and self._agg:
            result_df = map_df.groupby(self._groupby).agg(self._agg)
        else:
            result_df = map_df

        return result_df
    
    def _map(self, analysis, *args, **kwargs):
        results = []
        for item in self.group:
            try:
                if item.is_target_object(of=analysis):
                    results.append(analysis.apply_to(item, *args, **kwargs))
                else:
                    results.append(item.apply(analysis, *args, **kwargs))
            except Exception as exc:
                if self.error_policy == 'ignore':
                    #pass
                    results.append(None)
                elif self.error_policy == 'warn':
                    print(f"Failed on {item} with {exc}!")
                    results.append(None)
                else:
                    print(f"Failed on {item} with {exc}!")
                    raise exc                    

        return self._join_results(results)

    def _map_parallel(self, analysis, *args, n_processes=8, **kwargs):
        try:
            from multiprocess import Pool
        except ImportError:
            import warnings
            warnings.warn("It seems you do not have `multiprocess` installed, falling back to plain `multiprocessing`!"
                    " multiprocess works better for parallelization")
            from multiprocessing import Pool
        from functools import partial

        # TODO: Just noticed this behavior is different from _map, probably incorrect
        # This only allows parallelization across the fundamental Item of the analysis

        # TODO: error handling

        apply_analysis = partial(analysis.apply_to, *args, **kwargs)

        with Pool(n_processes) as p:
            results = p.map(apply_analysis, self.group)

        return self._join_results(results)

    def _join_results(self, results):
        keys = [item.label() for item in self.group]
        name = self.group.item_class.__name__

        new_results = []
        new_keys = []

        for k, r in zip(keys, results):
            if r is not None and not r.empty:
                new_results.append(r)
                new_keys.append(k)

        try:
            aligned_results = [new_results[0].align(r, axis=0)[1].dropna(how='all') for r in new_results]
            return pd.concat(aligned_results, keys=new_keys, names=[name])
            #return pd.concat([r for r in results if not r.empty], keys=keys, names=[name])
        except ValueError:
            return None
        except Exception as exc:
            print(f"Joining results failed due to {exc}")
            raise
            #try:
            #    return [pd.concat([res], keys=[key], names=[name])
            #        for res, key in zip(results, keys)]
            #except:
            #    return dict(zip(keys, results))

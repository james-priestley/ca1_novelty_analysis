import pandas as pd


@pd.api.extensions.register_dataframe_accessor("level")
@pd.api.extensions.register_series_accessor("level")
class LevelAccessor:
    """Syntactic sugar for working with multi-indexed dataframes.

    Examples
    ---------
    First import
    >>> from lab3.misc import level_accessor

    To get a single level from the index:
    >>> df.level['expt']

    To get multiple levels from the index
    >>> df.level[['expt', 'mouse']]

    To get rows where levels take certain values
    >>> df.level(mouse='mickey', expt='22001')

    To select rows in a more complex way
    >>> df.level('expt == 2200 | expt == 2201',
                 'duration < 600', mouse='mickey')

    For acceptable syntax refer to:
    https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.query.html
    """

    def __init__(self, pandas_obj):
        self._obj = pandas_obj

    def __getitem__(self, item):
        try:
            return self._obj.index.get_level_values(item)
        except KeyError:
            levels = self._obj.reset_index()[item]
            return pd.MultiIndex.from_frame(levels)

    def __call__(self, *query, **kwargs):
        query = list(query)
        for k, v in kwargs.items():
            try:
                # This may not work for namespacing reasons...
                assert (v[0] == "@") or (v[0] == v[-1] == '`')
                query.append(f'`{k}` == {v}')
            except (AssertionError, TypeError):
                query.append(f'`{k}` == "{v}"')

        try:
            return self._obj.query('(' + ') & ('.join(query) + ')' )
        except AttributeError:
            df = self._obj.to_frame()
            return df.query('(' + ') & ('.join(query) + ')')[self._obj.name]

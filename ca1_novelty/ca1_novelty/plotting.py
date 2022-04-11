import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def multi_heatmap(data, group, cmaps, ax=None, cbar=True, all_cbars=False,
                  **kws):
    """
    Parameters
    ----------
    data : 2d array
    group : 1d or 2d array
        Group label of the samples in data, which will be used to
        plot the different heatmaps.
        If 1d, it must match the length of one dimension of data.
        If 2d, it must be the same shape as data.
    cmaps : list or dict
        Names of heatmaps to use for each group label. If group is
        an array of ints, cmaps can be a list, otherwise it should be
        a dictionary.
    ax : matplotlib axis object, optional
        Specify an axis for plotting
    all_cbars : bool, optional
        Whether to plot all colorbars. Defaults to False
        (only the first is drawn)

    Additional keywords are passed directly to seaborn's heatmap function.
    """

    if ax is None:
        fig, ax = plt.subplots()

    if len(group.shape) == 1:
        if len(group) == data.shape[0]:
            group = np.stack([group] * data.shape[1]).T
        elif len(group) == data.shape[1]:
            group = np.stack([group] * data.shape[0])
        else:
            raise ValueError(
                "If mask is 1d, it must match the length of a dimension in "
                + "data")
    else:
        assert group.shape == data.shape

    for lidx, label in enumerate(np.unique(group)):
        sns.heatmap(data, mask=(group != label), cmap=cmaps[label],
                    cbar=cbar if lidx == 0 else all_cbars,  ax=ax, **kws)

    return ax

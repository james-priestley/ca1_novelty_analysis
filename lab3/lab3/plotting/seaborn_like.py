import numpy as np
import itertools as it
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

from utils import *

sns.set(context='talk', style='ticks')

def _slopegraph_helper(data, x, y, ax, xlim, hue_order=None, **kwargs):
    if hue_order is None:
        values = [df.to_frame(label).droplevel(x) for label, df in data[y].groupby(x)]
    else:
        values = [data.level(f'{x} == "{label}"')[y].to_frame(label).droplevel(x) for label in hue_order]
    assert len(values) == 2
    joined = values[0].join(values[1], how='inner')
    ax.plot(xlim, np.array(joined).T, **kwargs)
    
def slopeplot(data, x, y, ax=None, hue=None, order=None, hue_order=None, palette=None, **kwargs):
    """Plots a slopegraph 
    
    Parameters
    ----------
    x, y, hue : names of variables in ``data`` 
    data : dataframe, wide-form
        Wide-form dataframe where `x` is a subset of the columns
    ax : Axes instance, optional
        Axes to plot on
    palette : dict or palette, optional
        Colors to use by hue
    legend : bool
        Whether to include a legend
    **plt_kwargs 
    
    Returns
    -------
    ax : Axes instance
        Axes plotted on
    """
    if ax is None:
        fig, ax = plt.subplots()
    
    ticklabels = []
    
    if hue is None:
        _slopegraph_helper(data, x, y, ax=ax, xlim=(0,1), **kwargs)
    else:
        if order is None:
            hues = data.level[hue].unique()
        else:
            hues = order
        for i, hue_name in enumerate(hues): #enumerate(data.groupby(hue)):
            df = data.level(f'{hue} == "{hue_name}"')
            _slopegraph_helper(df, x, y, ax=ax, xlim=(i-1/6,i+1/6), hue_order=hue_order, **kwargs)
            ticklabels.append(hue_name)
            
        ax.set_xticks(np.arange(len(hues)))
    
    ax.set_xticklabels(ticklabels)

def slopegraph(data, x, ax=None, hue=None, palette=RANDOM_COLORS, legend=None, **plt_kwargs):
    """Plots a slopegraph 
    
    Parameters
    ----------
    x : array-like, two elements (does this work for more??)
        Columns of dataframe to plot as slopegraph
    data : dataframe, wide-form
        Wide-form dataframe where `x` is a subset of the columns
    ax : Axes instance, optional
        Axes to plot on
    hue : str, column of data
        Column to color data by
    palette : dict or palette, optional
        Colors to use by hue
    legend : bool
        Whether to include a legend
    **plt_kwargs 
    
    Returns
    -------
    ax : Axes instance
        Axes plotted on
    """
    
    if ax is None:
        fig, ax = plt.subplots()
    
    if hue is None:
        plot_data = np.array(data[x]).T
        ax.plot(plot_data, **plt_kwargs)
    else:
        legend_representatives = []
        
        for i, (hue_label, hue_data) in enumerate(data.groupby(hue)):
            if palette is None:
                palette = RANDOM_COLORS
            else:
                color = palette[hue_label]
            plot_data = np.array(hue_data[x]).T
            lines = ax.plot(plot_data, color=color, label=hue_label,
                    **plt_kwargs)
            legend_representatives.append((lines[0], hue_label))
            
    if legend:
        ax.legend(*zip(*legend_representatives))

    ax.set_xticks([0,1])
    ax.set_xlim([-0.25,1.25])
    ax.set_xticklabels(x)
    
    return ax

def histplot(data, x=None, bins=None, kde=False, shared_bins=True, hue=None, palette=RANDOM_COLORS, 
             ax=None, legend=False, **seaborn_kws):
    """Extends sns.distplot to work with dataframes, hues 
    
    Parameters
    ----------
    data : dataframe, wide-form
        Wide-form dataframe where `x` is a subset of the columns
    x : str, optional
        Column of dataframe to plot as histograms
    bins : int or bins, optional
        If int, all data will be plotted on same bins unless 
        `shared_bins` is False
    shared_bins : bool, optional
        Whether to plot different labels on same auto-generated bins. 
        Defaults to True.
    ax : Axes instance, optional
        Axes to plot on
    hue : str, column of data
        Column to color data by
    palette : dict or palette, optional
        Colors to use by hue
    legend : bool
        Whether to include a legend
    **seaborn_kws 
    
    Returns
    -------
    ax : Axes instance
        Axes plotted on
    """

    if x is None:
        hist_data = data
    else:
        hist_data = data[x]
        
    if bins is None:
        bins = 'auto'

    if shared_bins:
        bins = np.histogram_bin_edges(hist_data, bins=bins)
    
    if ax is None:
        fig, ax = plt.subplots()

    # if kde is True:
    #     seaborn_kws['norm_hist'] = True
    #     if 'kde_kws' in seaborn_kws:
    #         kde_kws = seaborn_kws['kde_kws']
    #     else:
    #         kde_kws = {}
    #     # if 'cumulative' in seaborn_kws:
    #     #     kde_kws['cumulative'] = seaborn_kws['cumulative']
    #     densityplot(data=data, x=x, hue=hue, palette=palette, 
    #                 ax=ax, **kde_kws)

    
    if hue is None:
        sns.distplot(hist_data, bins=bins, ax=ax, **seaborn_kws)
    else:
        for hue_key, hue_data in hist_data.groupby(hue):
            sns.distplot(hue_data, bins=bins, ax=ax, 
                         kde=kde,
                         color=palette[hue_key], 
                         label=hue_key,
                         **seaborn_kws)

    if legend:
        ax.legend()
    
    return ax

def densityplot(data, x=None, y=None, hue=None, palette=RANDOM_COLORS, 
    ax=None, legend=False, **seaborn_kws):
    """Extends sns.kdeplot to work with dataframes, hues 
    
    Parameters
    ----------
    data : dataframe, wide-form
        Wide-form dataframe where `x` is a subset of the columns
    x : str, optional
        Column of dataframe to plot as KDE
    y : str, optional
        Column of dataframe to plot as 2D KDE
    ax : Axes instance, optional
        Axes to plot on
    hue : str, column of data
        Column to color data by
    palette : dict or palette, optional
        Colors to use by hue
    legend : bool
        Whether to include a legend
    **plt_kwargs 
    
    Returns
    -------
    ax : Axes instance
        Axes plotted on
    """

    if x is None:
        hist_data = data
    else:
        hist_data = data[x]
        
    if ax is None:
        fig, ax = plt.subplots()
    
    if hue is None:
        sns.kdeplot(hist_data, ax=ax, **seaborn_kws)
    else:
        for hue_key, hue_data in hist_data.groupby(hue):
            sns.kdeplot(hue_data, ax=ax, 
                         color=palette[hue_key], 
                         label=hue_key,
                         **seaborn_kws)
    if legend:
        ax.legend()
    
    return ax


def traceplot(data, ax=None, lw=1, scalebar=False, **plt_kwargs):
    """Plots traces
    
    Parameters
    ----------
    data : dataframe, wide-form
        Wide-form dataframe of signals (raw, dfof, etc.)
    ax : Axes instance, optional
        Axes to plot on
    scalebar : bool
        Whether to plot a scalebar
    **plt_kwargs 
    
    Returns
    -------
    ax : Axes instance
        Axes plotted on
    """

    n_cells, n_samples = data.shape
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(20,10))
    
    plot_data = data.T - data.mean(axis=1)
    dy = (plot_data.max(axis=1) - plot_data.min(axis=1)).max()
    plot_data += dy*np.arange(n_cells) 
    
    ax.plot(plot_data, lw=lw, **plt_kwargs)
    if scalebar:
        plot_scalebar(ax, dx=n_samples/20, dy=1)    
    sns.despine(ax=ax, bottom=True, left=True)
    
    ax.set_yticks([])
    ax.set_xticks([])
    
    return ax

def trendline_plot(data, x=None, y=None, ax=None, hue=None, palette=None, # degree=1, # TODO
                    trendline_style={'ls':'--'}, show_eqn=False, show_r=False,
                   **seaborn_kws):
    """Extends sns.scatterplot with the option to show trendlines with equations
    
    Parameters
    ----------
    data : pd.DataFrame, wide-form
        Wide-form dataframe where `x` is a subset of the columns
    x : str, optional
        Column of dataframe to plot on the abscissa
    y : str, optional
        Column of dataframe to plot on the ordinate 
    ax : Axes instance, optional
        Axes to plot on
    hue : str, optional
        Column to color data by
    palette : dict or palette, optional
        Colors to use by hue
    show_eqn : bool, optional (default=False)
        Whether to show the trendline equation
    show_r : bool, optional (default=False)
        Whether to show the r^2 and p value 
    trendline_style : dict, optional
        Style for trendlines
    **seaborn_kws 
        Passed to sns.scatterplot
    
    Returns
    -------
    ax : Axes instance
        Axes plotted on
    """
    ax = sns.scatterplot(data=data, x=x, y=y, ax=ax, 
                         hue=hue, 
                         palette=palette, 
                         **seaborn_kws)
    if hue is not None:
        groups = data[hue].unique()
        for name, df in data.groupby(hue):
            color = palette[name] if palette else None
            
            m, b, r_val, p_val, std_err = stats.linregress(df[x], df[y])
            
            p = np.poly1d([m, b])
            
            eqn = f"y = {m:.2f}x + {b:.2f}"
            
            if show_r:
                eqn += f" (r^2={r_val**2:.2f}, p={p_val:.2f})"
            
            ax.plot(df[x], p(df[x]), c=color, label=eqn, **trendline_style)
    else:
        m, b, r_val, p_val, std_err = stats.linregress(data[x], data[y])
        p = np.poly1d([m, b])
        eqn = f"y = {m:.2f}x + {b:.2f}"
        if show_r:
            eqn += f" (r^2={r_val**2:.2f}, p={p_val:.2f})"
        ax.plot(data[x], p(data[x]), label=eqn, **trendline_style)
        
    if show_eqn:
        ax.legend()
        
    return ax

def _plot_single_arrowplot(df, ax, x=None, y=None, label=None, color='k', sort_values=None, **kwargs):
    if sort_values is not None:
        df = df.sort_values(sort_values)
    
    plot_data = df[[x,y]]
        
    for i in range(len(plot_data)-1):
        tail = plot_data.iloc[i]
        head = plot_data.iloc[i+1]
        arrow = ax.arrow(*tail, *(head-tail), fc=color, ec=color, length_includes_head=True,
                 label=label if i==0 else None, **kwargs)
        
    return arrow
        
    
def arrowplot(data, x=None, y=None, hue=None, palette=CYCLIC_COLORS, ax=None,
              sort_values=None, legend=False, head_scale=0.05, **kwargs):
    if ax is None:
        fig, ax = plt.subplots()
        
    xlim = [data[x].min(), data[x].max()]
    ylim =[data[y].min(), data[y].max()]
    
    head_width = head_scale*(xlim[1] - xlim[0])
    head_length = head_scale*(ylim[1] - ylim[0])    
    
    arrows = []
    labels = []
    
    if hue is not None:
        for label, df in data.groupby(hue):
            if len(df) > 1:
                arrow = _plot_single_arrowplot(df, ax=ax, x=x, y=y, label=label, color=palette[label], 
                                               sort_values=sort_values, head_width=head_width, 
                                               head_length=head_length, **kwargs)
                arrows.append(arrow)
                labels.append(label)
    else:
        _plot_single_arrowplot(df, ax=ax, x=x, y=y, 
                                sort_values=sort_values, head_width=head_width, 
                                head_length=head_length, **kwargs)

    ax.set_xlim(xlim)
    ax.set_ylim(ylim) 
    
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    
    if legend:
        ax.legend(arrows, labels)
    
    return ax

def parametric_plot(data, x, y, t, hue, **kwargs):
    df = data.sort_values(t)
    return sns.lineplot(data=data, x=x, y=y, hue=hue, **kwargs)

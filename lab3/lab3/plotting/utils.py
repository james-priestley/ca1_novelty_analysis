import pandas as pd 
import numpy as np
import itertools as it
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

def wide2long(df):
    return df.stack().to_frame(name="activity").reset_index()

def wrap(df, key, header=None):
    return pd.concat([df], keys=[key], names=[header])

class ColorCycler(dict):
    def __init__(self, palette=None, desat=None):
        self.colors = it.cycle(sns.color_palette(palette=palette, desat=desat))
        super().__init__({})
    
    def __getitem__(self, key):
        try:
            return super().__getitem__(key)
        except KeyError:
            color = next(self.colors)
            self[key] = color
            return color

class ArbitraryColors:
    def __init__(self, palette=None):
        """No guarantee these colors are aesthetically pleasing or
        do not repeat
        """
        if palette is None:
            self.palette = sns.color_palette()
        else:
            self.palette = palette
        self.n_colors = len(self.palette)
    
    def __getitem__(self, key):
        return self.palette[hash(key)%self.n_colors]
    
    def __call__(self, key):
        return self[key]

RANDOM_COLORS = ArbitraryColors(sns.color_palette('bright'))
CYCLIC_COLORS = ColorCycler()

def plot_scalebar(ax, dx=0, dy=0, x0=None, y0=None, color='k', lw=2, **kwargs):
    """Adds a scalebar to a plot

    Parameters
    ----------
    ax : Axes instance, optional
        Axes to plot on
    dx : float
        width of scalebar
    dy : float
        height of scalebar
    x0 : float, optional
        x-origin of scalebar (default: xlim)
    y0 : float, optional
        y-origin of scalebar (default: ylim)
    **plt_kwargs

    Returns
    -------
    ax : Axes instance
        Axes plotted on
    """

    if x0 is None:
        x0 = ax.get_xlim()[0]
    if y0 is None:
        y0 = ax.get_ylim()[0]

    # x-scalebar
    ax.plot([x0,x0+dx], [y0,y0], color=color, lw=lw, **kwargs)
    # y-scalebar
    ax.plot([x0,x0], [y0,y0+dy], color=color, lw=lw, **kwargs)

def plot_significant_value(ax, label, text="*", y_factor=0.02):
    tick_positions = dict(zip([t.get_text() for t in ax.get_xticklabels()],
                              ax.get_xticks()))
    x0 = tick_positions[label]

    ymin, ymax = ax.get_ylim()
    yrange = ymax - ymin

    y0 = ymax + yrange*y_factor
    ax.text(x0, y0, text)
    return ax

def plot_significant_comparison(ax, label1=None, label2=None, x0=None, x1=None, text="*", dx_factor=-0.05,
                                y_factor=0.02, dy_factor=0.05):

    if (x0 is None) and (x1 is None):
        tick_positions = dict(zip([t.get_text() for t in ax.get_xticklabels()],
                                  ax.get_xticks()))
        x0 = tick_positions[label1]
        x1 = tick_positions[label2]

    ymin, ymax = ax.get_ylim()
    yrange = ymax - ymin

    y0 = ymax + yrange*y_factor
    dy = yrange * dy_factor

    ax.plot([x0,x1], [y0,y0], c='k')
    ax.plot([x0,x0], [y0-dy, y0], c='k')
    ax.plot([x1,x1], [y0-dy, y0], c='k')

    midpt = (x0+x1)/2.
    xrange = x1-x0

    ax.text(midpt+xrange*dx_factor, y0, text)



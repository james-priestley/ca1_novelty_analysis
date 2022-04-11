"""Classes for representing place fields and place cells"""

from abc import ABCMeta, abstractmethod

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from seaborn import heatmap

from platz.util import find_first_n


class _PlaceField(metaclass=ABCMeta):

    parent = None

    @property
    @abstractmethod
    def field_boundaries(self):
        return self._field_boundaries

    @property
    @abstractmethod
    def tuning_map(self):
        pass

    @property
    @abstractmethod
    def mask(self):
        pass

    @abstractmethod
    def _todict(self):
        return {
            'cell_id': self.parent,
            'field_boundaries': self.field_boundaries,
        }


class PlaceField1D(_PlaceField):

    """
    Documentation

    Parameters
    ----------
    start, stop : int
    raster : array, (n_laps, n_bins)
    n : int, optional
        Number of active laps in window to qualify as the place field start
        Defaults to 3 laps
    window : int, optional
        Lap window size to consider when looking for consistent in-field
        firing. Defaults to 3 laps
    margin : int, optional
        Expand the place field boundaries by this many bins on each side
        (to account for possible spatial shifts due to BTSP, for example).
        Defaults to 5
    """

    def __init__(self, start, stop, raster, n=3, window=5, margin=5):

        self.n_bins = raster.shape[1]
        assert start < self.n_bins
        self.start = start
        assert stop <= self.n_bins
        self.stop = stop

        self.raster = np.array(raster)
        self.margin = margin
        self.first_lap = self._first_lap(n=n, window=window)

    @property
    def start_bin(self):
        return np.max([0, self.start - self.margin])

    @property
    def stop_bin(self):
        return np.min([self.stop + self.margin, self.n_bins])

    def _first_lap(self, n=3, window=3):
        thres = 0.05 * np.max(self.raster)
        return find_first_n(
            self.raster[:, self.start_bin:self.stop_bin].mean(axis=1) > thres,
            n, window
        )

    @property
    def tuning_map(self):
        return self.raster.mean(axis=0)

    @property
    def tuning_curve(self):
        """This is just an alias for `tuning_map`"""
        return self.tuning_map

    @property
    def field_boundaries(self):
        return [self.start, self.stop]

    @property
    def mask(self):
        m = np.zeros((self.n_bins))
        m[self.start:self.stop] = 1
        return m.astype('bool')

    def _todict(self):
        return super()._todict()

    def plot_raster(self, ax=None, cmap='Greys', border_color='blue',
                    border_style='--', **kwargs):

        if ax is None:
            ax = plt.subplot(111)

        heatmap(self.raster, ax=ax, cmap=cmap, **kwargs)
        plt.axvline(self.start, color=border_color, linestyle=border_style)
        plt.axvline(self.stop, color=border_color, linestyle=border_style)
        plt.xlabel('Position')
        plt.ylabel('Lap number')

        return ax


class PlaceField2D(_PlaceField):

    """This should be encoded as a mask over the environment"""

    def __init__(self, mask=None, coord=None, events=None, occupancy=None):

        if mask is not None:
            self._mask = mask
        elif coord is not None:
            # convert coordinates to mask, where to get shape of environment?
            raise NotImplementedError

    @property
    def mask(self):
        return self._mask

    @property
    def tuning_map(self):
        raise NotImplementedError

    def _todict(self):
        pass


class PlaceCell(list):

    """A PlaceCell is a container of _PlaceField-like objects"""

    def __init__(self, label, id=None, fields=[], tags=[]):
        # check out SIMA ROI class?
        self.label = label
        self.id = id
        self.tags = tags

        self.fields = fields

        # check that
        super().__init__(fields)

    @property
    def dataframe(self):
        # dataframe representation of the place fields
        pass

    def has_field(self):
        return bool(self)

    def _todict(self):
        pass


class PlacePopulation(list):

    """A PlacePopulation is a container of PlaceCell objects."""

    def __init__(self, cells):
        super().__init__(cells)

    @property
    def place_cells(self):
        return PlacePopulation([cell for cell in self if cell.has_field])

    @property
    def nonplace_cells(self):
        return PlacePopulation([cell for cell in self if ~cell.has_field])

    @property
    def dataframe(self):
        # Should
        return pd.concat([pc.dataframe for pc in self])

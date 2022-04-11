from abc import ABCMeta, abstractmethod

import numpy as np
import pandas as pd
from scipy.signal import gaussian, convolve

from ..base import BaseSignalAnalysis


class SpatialTuningStrategy(BaseSignalAnalysis, metaclass=ABCMeta):

    """Abstract base class for implementing strategies for calculating
    spatial tuning. Each derived class should implement a specific strategy for
    calculating spatial tuning curves for each neuron / lap, and specify
    a method for shuffling position and/or activity to generating null
    distributions.

    The details here are flexible: in the most straightforward scenarios,
    the input signals is a DataFrame with shape (n_rois, n_samples), but it
    could also be a structure containing discrete events (e.g. transients). The
    output of _calculate however should always result in a DataFrame as
    described below.

    Implement _smooth, _shuffle, and _calculate to subclass. Optionally
    implement calculate_rasters to return trial-by-trial tuning curves.
    """

    def calculate(self, signals, position, laps=None, max_position=None,
                  shuffle=False):
        """For each neuron, calculate the spatial tuning curve averaged over
        laps. This method returns a dataframe that is indexable like:

        >>> tuning_curves.loc[roi_label, position_number]

        Parameters
        ----------
        signals : pd.DataFrame
            Typically an array shape (n_rois, n_samples), keyed by the ROI
            label.
        position : array (n_samples)
            Array containing the integer position bin for every sample in
            signals. The first bin on the track should be zero.
        laps : array (n_samples)
            Array containing the lap id of ever sample in signals.
        max_position : int, optional
            The maximum position bin on the track. If None (default), this is
            assumed to be max(position). In most cases where positions are
            densely sampled over the recording, this argument can be ignored.
            After calculating tuning curves and before smoothing, it is used
            to populate any unsampled position bins in the tuning curve with
            NaNs.
        shuffle : bool, optional
            Whether to shuffle the position of each sample before computing
            tuning curves (i.e. for sampling the tuning curve null
            distribution)

        Returns
        -------
        tuning_curves : pd.DataFrame (n_rois, n_position_bins)
            Dataframe storing the spatial tuning curve for each cell.
        """

        if shuffle:
            signals, position, laps = self._shuffle(
                signals=signals, position=position, laps=laps)
        tuning_curves = self._calculate(signals, position, laps)

        # insert NaNs for missing position bins in the tuning curve
        if max_position is None:
            max_position = np.max(position)

        missing_bins = []
        for n in range(max_position):
            if n not in tuning_curves.columns:
                missing_bins.append(n)
                tuning_curves.loc[:, n] = np.nan
        tuning_curves = tuning_curves.sort_index(axis=1)

        if missing_bins:
            # interpolate to replace missing values for now before smoothing
            smooth_tuning_curves = self._smooth(
                tuning_curves.interpolate(
                    axis=1, method='linear', limit_direction='both'))
            # re-insert nans at missing values
            for n in missing_bins:
                smooth_tuning_curves.loc[:, n] = np.nan
            return smooth_tuning_curves
        else:
            return self._smooth(tuning_curves)

    # ----------------- Implement these methods to subclass ----------------- #

    @abstractmethod
    def _smooth(self, tuning_curves):
        """Apply smoothing to the resulting tuning curves

        Parameters
        ----------
        tuning_curves : pd.DataFrame (n_rois, n_position_bins)
            Dataframe storing the spatial tuning curve for each cell.

        Returns
        -------
        tuning_curves : pd.DataFrame (n_rois, n_position_bins)
            Dataframe storing the spatial tuning curve for each cell.
        """

    @abstractmethod
    def _shuffle(self, signals=None, position=None, laps=None):
        """Shuffle the data. For generating null tuning curves, you will
        typically want to shuffle either the signals themselves or their
        associated positions.

        Parameters
        ----------
        signals : array (n_rois, n_samples), optional
            Typically an array shape (n_rois, n_samples), keyed by the ROI
            label.
        position : array (n_samples), optional
            Array containing the integer position bin for every sample in
            signals
        laps : array (n_samples), optional
            Array containing the lap id of ever sample in signals. Depending on
            your shuffling method, you may or many not need the laps variable.
        """

    @abstractmethod
    def _calculate(self, signals, position, laps=None):
        """Calculate spatial tuning for each neuron. This should return a
        dataframe indexable like:

        >>> tuning_curves.loc[roi_label, position_number]

        Parameters
        ----------
        signals : array (n_rois, n_samples)
            Typically an array shape (n_rois, n_samples), keyed by the ROI
            label.
        position : array (n_samples)
            Array containing the integer position bin for every sample in
            signals
        laps : array (n_samples), optional
            Array containing the lap id of ever sample in signals. Depending on
            your tuning calculation method, you may or many not need the laps
            variable.
        """

    # ------------------------------ Optional ------------------------------- #

    def calculate_rasters(self, signals, position, laps, smooth=True):
        """This should return a multi-index dataframe with the spatial tuning
        for each lap, indexible like:

        >>> rasters.loc[(roi_label, lap_number), position_number]

        Depending on the algorithm, it may or may not be performant to use
        calculate_rasters when implementing _calculate. It is still useful to
        implement anyway, for example, to provide trial-level tuning for
        plotting.

        Parameters
        ----------
        signals : array (n_rois, n_samples)
            Typically an array shape (n_rois, n_samples), keyed by the ROI
            label.
        position : array (n_samples)
            Array containing the integer position bin for every sample in
            signals
        laps : array (n_samples)
            Array containing the lap id of ever sample in signals. Depending on
            your tuning calculation method, you may or many not need the laps
            variable.
        smooth : bool, optional
            Whether to apply smoothing to each lap. Defaults to True
        """
        raise NotImplementedError


class SimpleSpatialTuning(SpatialTuningStrategy):

    """A simple spatial tuning strategy. For each neuron, we compute its
    average activity across samples/laps for each position. The tuning curves
    are optionally smoothed with a Gaussian kernel. Since we are computing
    the average activity across samples for a given position, this implicitly
    normalizes for occupancy.

    Shuffled curves are generated by circularly shifting the position of
    samples independently within each lap.

    Parameters
    ----------
    sigma : float, optional
        Standard deviation of the gaussian smoothing kernel applied to the
        tuning curves. By default, no smoothing is applied.

    Examples
    --------
    >>> # generate some fake data
    >>> signals = pd.DataFrame(np.random.normal(0, 1, (10, 1000)))
    >>> laps = np.concatenate([np.ones(100) * lap for lap in range(10)])
    >>> position = np.concatenate([np.arange(100)] * 10)
    >>>
    >>> # compute the true tuning curves
    >>> strategy = SimpleSpatialTuning(sigma=2)
    >>> tuning_curves = strategy.calculate(
    >>>     signals, position, laps, shuffle=False)
    >>>
    >>> # get a set of null tuning curves via circular shuffle
    >>> tuning_curves = strategy.calculate(
    >>>     signals, position, laps, shuffle=True)
    """

    def __init__(self, sigma=None):
        self.sigma = sigma
        if self.sigma:
            kernel = gaussian(sigma * 8, sigma)
            self._k = kernel / kernel.sum()

    def _smooth(self, tuning_curves, random_state=None):
        """Apply Gaussian smoothing across position bins"""
        if self.sigma is None:
            return tuning_curves
        else:
            tuning_curves.iloc[:, :] = np.apply_along_axis(
                convolve, 1, tuning_curves, self._k, mode='same')
            return tuning_curves

    def _shuffle(self, signals=None, position=None, laps=None):
        """Applies a circular shuffle. On ever lap, shuffle the position
        samples by a random number of samples. signals input is ignored."""
        shuffled_position = np.concatenate([np.roll(
            position[(laps == l)], np.random.randint((laps == l).sum()))
                               for l in np.unique(laps)])
        return signals, shuffled_position, laps

    def _calculate(self, signals, position, laps=None):
        """For ROI, simply calculate the average of all samples within each
        position.

        Parameters
        ----------
        signals : pd.DataFrame
        position : array
        laps : ignored
        """

        columns = pd.MultiIndex.from_arrays([position, signals.columns],
                                            names=['position', 'time'])
        signals = pd.DataFrame(signals.values, columns=columns,
                               index=pd.Index(signals.index, name='roi_label'))
        return signals.T.groupby('position').mean().T

    def calculate_rasters(self, signals, position, laps, smooth=True,
                          fill_nans=True):
        """doc string

        Parameters
        ----------
        signals : pd.DataFrame
        position : array
        laps : array
        fill_nans : bool
        """

        columns = pd.MultiIndex.from_arrays([position, laps],
                                            names=['position', 'lap'])
        signals = pd.DataFrame(signals.values, columns=columns,
                               index=pd.Index(signals.index, name='roi_label'))
        rasters = signals.T.groupby(
            ['position', 'lap']).mean().unstack('lap').T

        if fill_nans:
            rasters = rasters.fillna(0)
        if smooth:
            rasters = self._smooth(rasters)
        return rasters


class CircularMixin:

    """Mixin class that modifies the functionality of _smooth, in order to
    account for the circularity of the track. In derived classes, this should
    be the first parent class.
    """

    def _smooth(self, tuning_curves):
        n_position_bins = tuning_curves.shape[1]
        tuning_curves = pd.concat([tuning_curves] * 3, axis=1)
        smooth_curves = super()._smooth(tuning_curves)
        return smooth_curves.iloc[:, n_position_bins:(n_position_bins * 2)]

    def _trial_smooth(self, rasters):
        n_position_bins = rasters.shape[1]

        # iterate over ROIs and append the previous and next lap to the left
        # and right of each trace respectively
        trial_curves = []
        for label, roi in rasters.groupby('roi_label'):
            previous_lap = np.vstack([np.zeros(roi.shape[1]),
                                      roi.values[:-1, :]])
            next_lap = np.vstack([roi.values[1:, :],
                                  np.zeros(roi.shape[1])])
            trial_curves.append(pd.DataFrame(
                np.hstack([previous_lap, roi.values, next_lap]),
                index=roi.index))

        # smooth
        smooth_curves = super()._smooth(pd.concat(trial_curves))
        smooth_curves = \
            smooth_curves.iloc[:, n_position_bins:(n_position_bins * 2)]
        smooth_curves.columns = rasters.columns
        return smooth_curves

    def calculate_rasters(self, *args, smooth=True, **kwargs):
        rasters = super().calculate_rasters(*args, smooth=False, **kwargs)
        if smooth:
            return self._trial_smooth(rasters)
        else:
            return rasters


class CircularSimpleSpatialTuning(CircularMixin, SimpleSpatialTuning):

    """Tuning strategy identical to SimpleSpatialTuning, but with smoothing
    that accounts for the edge effects of circular tracks.

    Parameters
    ----------
    sigma : float, optional
        Standard deviation of the gaussian smoothing kernel applied to the
        tuning curves. By default, no smoothing is applied.
    """

    pass


class TransientSpatialTuning(SpatialTuningStrategy):

    """TODO : the format of this will depend on the manner in which transients
    are stored.
    """

    pass


class CircularTransientSpatialTuning(CircularMixin, TransientSpatialTuning):

    pass

from abc import ABCMeta, abstractmethod
from itertools import repeat
from multiprocessing import Pool

import numpy as np
import pandas as pd

from lab3.analysis.misc import find_intervals
from .spatial_tuning import SpatialTuningStrategy, CircularMixin
from ..base import BaseSignalAnalysis


class PlaceFieldMeta(ABCMeta):

    """Custom metaclass for place field strategies, to force the user to
    implement certain attributes."""

    abstract_attributes = []

    def __call__(self, *args, **kwargs):
        obj = super().__call__(*args, **kwargs)
        for attr_name in obj.abstract_attributes:
            if not getattr(obj, attr_name):
                raise ValueError(f'Required attribute ({attr_name}) not set')
            if attr_name == 'tuning_strategy':
                ts = getattr(obj, attr_name)
                assert isinstance(ts, SpatialTuningStrategy), \
                    f"{ts} is not a valid SpatialTuningStrategy"
        return obj


class PlaceFieldStrategy(BaseSignalAnalysis, metaclass=PlaceFieldMeta):

    """Abstract base class for implementing place field detection algorithms.
    """

    abstract_attributes = ['tuning_strategy'] #, 'metric_functions']

    @property
    def circular(self):
        """Returns True if self.tuning_strategy implies a circular track.
        In this case we allow place fields to wrap around the ends of the
        track."""
        return isinstance(self.tuning_strategy, CircularMixin)

    def detect(self, signals, position, laps=None, sample_filter=None):
        """This is the main method for place field detection.

        Parameters
        ----------
        signals : pd.DataFrame
            Typically an array shape (n_rois, n_samples), keyed by the ROI
            label. Exact form of input depends on the tuning strategy used.
        position : array (n_samples)
            Array containing the integer position bin for every sample in
            signals
        laps : array (n_samples)
            Array containing the lap id of ever sample in signals.
        sample_filter : array (n_samples)
            Boolean array. Sample indices that are False are ignored during
            detection.

        Returns
        -------
        place_fields : pd.DataFrame (variable length)
            This should be a multi-index dataframe with top index roi_label
            and second index counting each place field. The columns should
            include...?
        """

        if sample_filter is not None:
            signals = signals.loc[:, sample_filter]
            position = position[sample_filter]
            laps = laps[sample_filter]

        return self._detect(signals, position, laps)

    def apply_metrics(self):
        try:
            return self._calculate_metrics(self)
        except AttributeError:
            raise Exception("You must detect place fields before applying "
                            + "metrics")

    # ----------------- Implement these methods to subclass ----------------- #

    @abstractmethod
    def _detect(self):
        """doc string"""
        pass

    @abstractmethod
    def _calculate_metrics(self):
        """doc string"""
        pass


def _shuffle(inputs):
    """Multiprocessing helper function"""
    np.random.seed()
    strategy_instance, signals, position, lap = inputs
    return strategy_instance.calculate(signals, position, lap, shuffle=True)


class SimplePFDetector(PlaceFieldStrategy):

    """Detect place fields by comparing the empirical tuning curve to a null
    distribution of tuning curves. Candidate place fields are spatial intervals
    where a neuron's firing rate exceeds the null distribution threshold,
    determined for the desired p-value. Place fields are then refined based
    on some simple secondary criteria: minimum and maximum allowable place
    field sizes, and a minimum number of trials that the cell fires within
    the candidate field.

    The manner in which the null distribution is sampled depends on the choice
    of SpatialTuningStrategy, which provides an additional level of
    customization to the detection beyond tuning the secondary parameters. As
    a result, this class is highly flexible and many detection schemes are
    possible.

    Parameters
    ----------
    tuning_strategy : SpatialTuningStrategy
        Instance of a SpatialTuningStrategy to be used for calculating
        tuning curves
    num_shuffles : int, optional
        How many shuffle iterations to run for generating shuffled tuning
        curves.
    pval_threshold : float, optional
        Threshold for identifying putative place fields, by comparing the
        spatial tuning curve at each bin to the null distribution. Defaults
        to 0.05 (i.e., firing rate must exceed 95% of null tuning curves at
        that bin)
    min_field_width, max_field_width : float, optional
        Minimum and maximum allowable widths to accept place fields, as a
        fraction of the track. Defaults to 0 and 1 respectively (no limits)
    min_trials_active : int, optional
        Only retain place fields wherein the cell fired on at least this many
        laps. Defaults to None (all fields are retained). Requires a
        `tuning_strategy` that implements `calculate_rasters`.
    n_processes : int, optional
        Number of workers to spawn in the parallel pool. Defaults to no
        parallelization

    Attributes
    ----------
    tuning_curves : pd.DataFrame (num_rois, num_position_bins)
    null_curves : pd.DataFrame (n_rois * num_shuffles, num_position_bins)
        Multi-indexable as null_curves.loc[(roi_label, shuffle_count)]
    threshold_curves : pd.DataFrame (num_rois, num_position_bins)
        The threshold activity rate at each bin to reach the desired
        significance level
    place_fields : pd.DataFrame (variable length)
        Multi-indexable as place_fields.loc[(roi_label, place_field_count)]
        Not all ROIs will have place fields, and some ROIs may have multiple
        fields

    Examples
    --------
    >>> # generate some fake data
    >>> signals = pd.DataFrame(np.random.normal(0, 1, (10, 1000)))
    >>> laps = np.concatenate([np.ones(100) * lap for lap in range(10)])
    >>> position = np.concatenate([np.arange(100)] * 10)
    >>>
    >>> # choose a tuning strategy
    >>> from lab3.analysis.spatial import SimpleSpatialTuning
    >>> tuning_strategy = SimpleSpatialTuning(sigma=3)
    >>>
    >>> # create detector object and detect place fields
    >>> detector = SimplePFDetector(tuning_strategy, pval_threshold=0.01,
    >>>                             min_field_width=0.05, max_field_width=0.5,
    >>>                             min_trials_active=5, num_shuffles=1000,
    >>>                             n_processes=8)
    >>> place_fields = detector.detect(signals, position, laps=laps)

    If the track is circular, we just need to pass a circular tuning strategy,
    and the detector will account for the boundary conditions.

    >>> from lab3.analysis.spatial import SimpleSpatialTuning
    >>> tuning_strategy = CircularSimpleSpatialTuning(sigma=3)
    >>> place_fields = SimplePFDetector(tuning_strategy).detect(
    >>>     signals, position, laps=laps)
    """


    def __init__(self, tuning_strategy, num_shuffles=10, pval_threshold=0.05,
                 min_field_width=0, max_field_width=1, min_trials_active=None,
                 n_processes=1):
        self.tuning_strategy = tuning_strategy
        self.num_shuffles = num_shuffles
        self.pval_threshold = pval_threshold
        self.min_field_width = min_field_width
        self.max_field_width = max_field_width
        self.min_trials_active = min_trials_active
        self.n_processes = n_processes

    def _get_nulls(self, signals, position, laps):
        """Use the inputs and self.tuning_strategy to sample from the null
        distribution of shuffle tuning curves"""

        # sample from null distribution
        iterator = repeat((self.tuning_strategy, signals, position, laps),
                          self.num_shuffles)
        if self.n_processes > 1:
            p = Pool(self.n_processes)
            null_curves = p.map(_shuffle, iterator)
            p.close()
            p.join()
        else:
            null_curves = map(_shuffle, iterator)

        # store nulls in multi-index dataframe
        null_curves = pd.concat(
                {n_idx: n for n_idx, n in enumerate(null_curves)},
                names=['shuffle', 'roi_label']
            ).swaplevel(0, 1).sort_index()

        # store as attributes
        self.null_curves = null_curves

        return null_curves

    def _get_thresholds(self):
        """Determine the threshold activity level at each spatial bin to
        establish significant spatial modulation at that location at
        p < self.pval_threshold"""

        # calculate thresholds
        thresholds = self.null_curves.groupby('roi_label').apply(
            np.percentile, axis=0, q=100 * (1 - self.pval_threshold))
        thresholds = pd.DataFrame.from_dict(
            dict(zip(thresholds.index, thresholds.values))).T
        thresholds.index.name = 'roi_label'
        thresholds.columns.name = 'position'

        # store as attributes
        self.threshold_curves = thresholds

        return thresholds

    def _find_fields(self, tuning_curves=None, threshold_curves=None):
        """Using self.tuning_curves and self.thresholds, evaluate for each
        neuron the spatial intervals along the tuning curve that exceed
        the significance threshold. Threshold-crossing intervals that meet
        all other criteria are retained as place fields"""

        if tuning_curves is None:
            tuning_curves = self.tuning_curves
        if threshold_curves is None:
            threshold_curves = self.threshold_curves

        num_bins = tuning_curves.shape[1]
        min_width = num_bins * self.min_field_width
        max_width = num_bins * self.max_field_width

        place_fields = []
        for label, tuning in tuning_curves.iterrows():

            suprathres_positions = tuning > threshold_curves.loc[label]
            if all(suprathres_positions == 0):
                continue  # there are no suprathreshold events
            else:
                # find spatial intervals that exceed the null and evaluate
                # the other conditions to decide if it's a place field or not
                count = 0
                for start, stop in find_intervals(
                        suprathres_positions, circular=self.circular):

                    # apply conditions
                    field_width = stop - start if stop > start \
                            else stop + num_bins - start

                    if field_width < min_width:
                        continue
                    if field_width > max_width:
                        continue
                    if self.min_trials_active is not None:
                        if stop > start:
                            n_trials_active = self.rasters.loc[
                                    label].loc[:, start:(stop-1)].mean(
                                    axis=1).gt(0).sum()
                        else:
                            # handle the circular wrap-around
                            left_edge = self.rasters.loc[
                                    label, 0:(stop-1)].mean(axis=1).gt(0)
                            right_edge = self.rasters.loc[
                                    label, start:].mean(axis=1).gt(0)
                            n_trials_active = (left_edge + right_edge).sum()
                        if n_trials_active < self.min_trials_active:
                            continue

                    place_fields.append({'roi_label': label,
                                         'field_count': count,
                                         'start_bin': start,
                                         'stop_bin': stop})
                    count += 1

        self.place_fields = pd.DataFrame(place_fields, columns=[
                'roi_label', 'field_count', 'start_bin', 'stop_bin']
            ).set_index(['roi_label', 'field_count']).sort_index()

        return self.place_fields

    def _detect(self, signals, position, laps):
        """Detect place fields from the input signals. Along the way, this
        assigns the attributes self.tuning_curves, self.threshold_curves,
        self.null_curves, self.place_fields, and optionally, self.rasters."""
        # get observed tuning
        self.tuning_curves = self.tuning_strategy.calculate(
            signals, position, laps)
        if self.min_trials_active is not None:
            self.rasters = self.tuning_strategy.calculate_rasters(
                signals, position, laps)

        # generate null distribution and calculate thresholds
        self._get_nulls(signals, position, laps)
        self._get_thresholds()

        # detect place fields
        return self._find_fields()

    def _calculate_metrics(self):
        """doc string"""
        pass


class SignedPFDetector(SimplePFDetector):

    """Extension of SimplePFDetector that permits the detection of positive or
    negative ("anti") place fields, which can be useful for analyzing the
    activity of interneurons.

    """

    def _get_thresholds(self):
        """Basically we need to establish an upper and lower threshold
        at half the p-value on each side"""

        raise NotImplementedError

    def _find_fields(self):
        """"""

        pos_fields = super().find_fields(
            tuning_curves=self.tuning_curves,
            threshold_curves=self.pos_threshold_curves
        )
        pos_fields['sign'] = 'positive'

        neg_fields = super().find_fields(
            tuning_curves=self.tuning_curves * - 1,
            threshold_curves=self.neg_threshold_curves * -1
        )
        neg_fields['sign'] = 'negative'

        self.place_fields = pd.concat([pos_fields, neg_fields]).sort_index()

        return self.place_fields


class PeakPFDetector(PlaceFieldStrategy):

    """Detect place fields by finding cells with high peak-to-mean firing
    rate gain in the place field tuning curve, with some additional
    constraints. A simple detector that does not require shuffling.

    Parameters
    ----------
    peak_gain : float, optional
        Minimum peak-to-mean firing rate gain, computed from the spatial tuning
        curve, for each bin to qualify as being in a place field. Defaults to 3
    min_rate : float, optional
        Minimum mean activity rate. Defaults to 0 (no minimum)
    min_field_width, max_field_width : float, optional
        Minimum and maximum widths of place field, as fraction of belt length.
        Defaults to 0.03 and 0.5 respectively
    """

    def __init__(self, tuning_strategy, peak_gain=3, min_rate=0,
                 min_field_width=0.05, max_field_width=1):

        raise NotImplementedError

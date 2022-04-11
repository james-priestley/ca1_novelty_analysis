"""Classes for modeling each experiment type"""

import os
# from copy import deepcopy

import numpy as np
import pandas as pd

from platz.tuning import SimpleSpatialTuning
from platz.detect import SimplePFDetector

from lab3.experiment.virtual import VirtualImagingExperiment
from lab3.experiment.group import ExperimentGroup, Cohort
from lab3.signal.decomposition import LinearDynamicalSystem

from ca1_novelty.place_cells import PlaceField


class AxonMixin:

    """"""

    def calculate_lds(self, from_type='dfof', n_components=1, **kwargs):
        """Apply dimensionality reduction by fitting a
        linear dynamical system

        Parameters
        ----------
        from_type : str, optional
            Signal type to use as the observed variables in the LDS.
            Defaults to 'dfof'
        n_components : int, optional
            Dimensionality of the latent space. Defaults to 1
        """
        strategy = LinearDynamicalSystem(n_components=1)
        return self.calculate_signal(
            strategy, from_type='dfof', to_type='lds', **kwargs)


class AxonExperiment(AxonMixin, VirtualImagingExperiment):

    pass


class SpatialMixin:

    """Mixin for experiments with spatial tuning analysis."""

    def set_tuning_strategy(self, sigma=3):
        self._tuning_strategy = SimpleSpatialTuning(sigma=sigma)
        return self

    def set_pf_detector(self, num_shuffles=1000, pval_threshold=0.05,
                        max_field_width=0.33, min_trials_active=15,
                        n_processes=8):
        self._pf_detector = SimplePFDetector(
            self.tuning_strategy,
            num_shuffles=num_shuffles,
            pval_threshold=pval_threshold,
            max_field_width=max_field_width,
            min_trials_active=min_trials_active,
            n_processes=n_processes)

    @property
    def tuning_strategy(self):
        """Returns the spatial tuning calculator"""
        try:
            return self._tuning_strategy
        except AttributeError:
            self.set_tuning_strategy()
            return self._tuning_strategy

    @property
    def pf_detector(self):
        """Return the place field detector"""
        try:
            return self._pf_detector
        except AttributeError:
            self.set_pf_detector()
            return self._pf_detector

    def find_place_fields(self):
        # don't really need this, unless we want a non-block specific one
        # see ContextMixin
        raise NotImplementedError

    def get_tuning_inputs(self, valid_samples=True, num_bins=100, beh_kws={},
                          signal_type='spikes', binarize=True, signal_kws={}):
        """Convenience function to get the inputs needed for calculating
        spatial tuning.

        Parameters
        ----------
        valid_samples : bool, optional
            Only include samples outside of ITI periods, where the animal is
            running with velocity > 5 cm/sec. Defaults to True
        num_bins : int, optional
            Resolution of spatial tuning curves. Defaults to 100 bins
        beh_kws : dict, optional
            Additional behavior formatting parameters.
            See `BehaviorExperiment.format_behavior_data`
        signal_type : str, optional
            Defaults to 'spikes'.
        binarize : bool, optional
            Whether to binarize signal, if signal_type is 'spikes'. Defaults to
            True.
        signal_kws : dict, optional
            Additional signal parameters.
            See `ImagingMixin.signals`.
        """

        signals = self.signals(signal_type=signal_type, **signal_kws)
        if (signal_type == 'spikes') and binarize:
            signals = signals.gt(0)

        position = self.discrete_position(num_bins=num_bins, **beh_kws)
        laps = self.laps(**beh_kws)

        if valid_samples:
            valid = self.valid_samples(**beh_kws)
            signals = signals.loc[:, valid]
            position = position[valid]
            laps = laps[valid]

        return signals, position, laps

    def rasters(self, smooth=True, **tuning_kws):
        """Returns a dataframe of spatial raster plots. The dataframe is
        organized with a multi-index of roi labels and lap numbers on the rows,
        and positions along the columns. This can be sliced like:

        >>> rasters.loc[pd.IndexSlice[roi_label, lapA:lapB], posA:posB]

        Parameters
        ----------
        smooth : bool, optional
            Whether to apply smoothing to activity on each lap, as defined by
            `self.tuning_strategy`. Defaults to True

        Additional keyword arguments are passed directly to
        `self.get_tuning_inputs`
        """
        try:
            # use place field detector to get rasters
            raise AttributeError
        except AttributeError:
            # no place fields calculated yet, so just use tuning strategy
            signals, position, laps = self.get_tuning_inputs(**tuning_kws)

            return self.tuning_strategy.calculate_rasters(
                signals, position, laps, smooth=smooth)

    def tuning_curves(self, smooth=True, **tuning_kws):
        """Returns a dataframe of the lap-averaged tuning curves, indexed by
        roi labels with position along the columns.

        Parameters
        ----------
        smooth : bool, optional
            Whether to apply smoothing to activity on each lap, as defined by
            `self.tuning_strategy`. Defaults to True

        Additional keyword arguments are passed directly to
        `self.get_tuning_inputs`
        """
        try:
            # use place field detector to get rasters
            raise AttributeError
        except AttributeError:
            # no place fields calculated yet, so just use tuning strategy
            signals, position, _ = self.get_tuning_inputs(**tuning_kws)

            return self.tuning_strategy.calculate(
                signals, position)  # , smooth=smooth)


class PlaceCellMixin(SpatialMixin):

    # Do we need this?

    pass


class ContextMixin:

    """Mixin for experiments with VR context switches"""

    @property
    def is_novel(self):
        return bool(self._trial_info['novel'])

    @property
    def day(self):
        return self._trial_info['day']

    @property
    def set_id(self):
        return 0 if 'ctx_2' in self.vr_context_names else 1

    def iter_blocks(self, **kwargs):
        """Generator that yields the context label and start/stop laps for
        each context block.

        Parameters
        ----------
        Keyword arguments are passed directly to
        `BehaviorExperiment.format_behavior_data`

        Yields
        ------
        context : str
            Name of the context for the current block
        (start, stop) : tuple of ints
            Start and stop laps of the current block. Note stop is the first
            lap *not* in the current block, for easy slicing.
        """
        start = 0
        lap_ids = self.lap_vr_context_id(**kwargs)
        current_ctx = lap_ids[0]
        for lap_num, lap_ctx in enumerate(lap_ids):
            final_lap = lap_num == len(lap_ids) - 1
            if (lap_ctx != current_ctx) | final_lap:
                stop = lap_num if not final_lap else lap_num + 1
                yield (current_ctx, [start, stop])

                current_ctx = lap_ctx
                start = lap_num

    def iter_block_samples(self, valid_samples=False, **kwargs):
        """Generator that yields the context label and start/stop sample
        indices for each context block.

        Parameters
        ----------
        valid_samples : bool, optional
            Only include samples outside of ITI periods, where the animal is
            running with velocity > 5 cm/sec. Defaults to False
        Keyword arguments are passed directly to
        `BehaviorExperiment.format_behavior_data`

        Yields
        ------
        context : str
            Name of the context for the current block
        (start, stop) : tuple of ints
            Start and stop sample indices of the current block. Note stop is
            the first sample *not* in the current block, for easy slicing.
        """
        # TODO this can probably be simpler
        sample_ids = self.sample_vr_context_id(**kwargs)
        if valid_samples:
            sample_ids = sample_ids[self.valid_samples(**kwargs)]

        start = 0
        current_ctx = sample_ids[0]
        for sample_num, sample_ctx in enumerate(sample_ids):
            if sample_ctx is None:
                continue
            final_sample = sample_num == len(sample_ids) - 1
            if (sample_ctx != current_ctx) | final_sample:
                stop = sample_num if not final_sample else sample_num + 1
                yield current_ctx, [start, stop]

                current_ctx = sample_ctx
                start = sample_num

    def context_blocks(self, **kwargs):
        """See `self.iter_blocks`.

        Returns
        -------
        block_info : list of tuples
            Returns context names and lap boundaries for all blocks in a list
        """
        block_info = self.iter_blocks(**kwargs)
        return list(block_info)

    def block_rasters(self, smooth=True, **tuning_kws):
        """Returns a dataframe of spatial raster plots, additionally labeled by
        the block number of each trial. The dataframe is organized with a
        multi-index of roi labels, block numbers, context, and trials on the
        rows, and positions along the columns. This can be sliced like:

        >>> rasters.loc[
        ...     pd.IndexSlice[roi_label, block_number, context, lapA:lapB],
        ...     posA:posB]

        Parameters
        ----------
        smooth : bool, optional
            Whether to apply smoothing to activity on each lap, as defined by
            `self.tuning_strategy`. Defaults to True

        Additional keyword arguments are passed directly to
        `self.get_tuning_inputs`
        """

        rasters = self.rasters(smooth=smooth, **tuning_kws)

        block_rasters = {}
        for block_num, (context, (block_start, block_stop)) \
                in enumerate(self.iter_blocks()):

            # pandas slicing is inclusive so subtract 1 from block_stop
            block_stop -= 1
            block_rasters[(block_num, context)] = rasters.loc[
                pd.IndexSlice[:, block_start:block_stop], :]

        block_rasters = pd.concat(
            block_rasters, names=['block_num', 'context']).reorder_levels(
                [2, 0, 1, 3]).sort_index()
        return block_rasters

    def block_tuning_curves(self, smooth=True, first_n_trials=None,
                            **tuning_kws):
        """Returns a dataframe of the lap-averaged tuning curves, computed
        separately for each trial block. The dataframe is organized with a
        multi-index of roi labels, block numbers, and context on the rows, and
        positions along the columns. This can be sliced like:

        >>> rasters.loc[
        ...     pd.IndexSlice[roi_label, block_number, context],
        ...     posA:posB]

        Parameters
        ----------
        smooth : bool, optional
            Whether to apply smoothing to activity on each lap, as defined by
            `self.tuning_strategy`. Defaults to True
        first_n_trials : int, optional
            If passed, compute the tuning curves using only the first n trials
            in each block. Defaults to None (all trials are used).

        Additional keyword arguments are passed directly to
        `self.get_tuning_inputs`
        """
        levels = ['roi_label', 'block_num', 'context']
        block_rasters = self.block_rasters(smooth=smooth, **tuning_kws)

        block_grouper = block_rasters.groupby(levels)

        if first_n_trials is None:
            block_curves = block_grouper.aggregate(
                np.nanmean, axis=0)
        else:
            truncated_block_rasters = pd.concat(
                [b.iloc[0:first_n_trials] for _, b in block_grouper])
            block_curves = truncated_block_rasters.groupby(
                levels).aggregate(np.nanmean, axis=0)

        return block_curves

    def place_fields_file(self, mode='r', **kwargs):
        return pd.HDFStore(os.path.join(self.sima_path, "place_fields.h5"),
                           mode=mode, **kwargs)

    # def _detect_block_place_fields(self, ):

    def detect_block_place_fields(self, signal_type='spikes', binarize=True,
                                  tuning_kws={}, overwrite=False,
                                  **detector_kws):
        """Returns a dataframe of place field locations, detected separately
        for each trial block. The dataframe is organized with a
        multi-index of roi labels, block numbers, context, and field count on
        the rows. This can be sliced like:

        >>> place_fields.loc[
        ...     pd.IndexSlice[roi_label, block_number, context, field_count],
        ...     ]

        Parameters
        ----------
        signal_type : str, optional
        binarize : bool, optional
        tuning_kws : dict
            Additional tuning parameters, specified as a dictionary.
             See `SpatialMixin.get_tuning_inputs` for options.
        num_shuffles : int, optional
            Defaults to 1000
        pval_threshold : float (0, 1]
            Defaults to 0.05
        max_field_width : float (0, 1]
            As fraction of belt. Defaults to 0.33
        min_trials_active : int, optional
            Defaults to 10.
        n_processes : int, optional
            Defaults to 8, because Ana Sofia says so
        """

        # TODO save results in an h5 to disk, but then we need to handle
        # label/channel info?
        # should also save thresholds and null_curves (expensive)
        if not overwrite:
            try:
                with self.place_fields_file() as pf_file:
                    place_fields = pf_file.get("place_fields")
                    null_curves = pf_file.get("null_curves")
                    threshold_curves = pf_file.get("threshold_curves")

                return place_fields, null_curves, threshold_curves
            except (OSError, FileNotFoundError, KeyError):
                print("Place fields not found. Running detection...")
                pass

        signals, position, laps = self.get_tuning_inputs(
            signal_type=signal_type, binarize=binarize, **tuning_kws
        )

        place_fields, null_curves, threshold_curves = [], [], []
        for block_num, (ctx, (start, stop)) in enumerate(
                self.iter_block_samples(valid_samples=True)):

            block_pfs = self.pf_detector.detect(
                signals.iloc[:, start:stop],
                position[start:stop],
                laps[start:stop]
            )

            # add some additional metadata for the index
            block_pfs['context'] = ctx
            block_pfs['block_num'] = block_num

            block_null_curves = self.pf_detector.null_curves
            block_null_curves['context'] = ctx
            block_null_curves['block_num'] = block_num

            block_threshold_curves = self.pf_detector.threshold_curves
            block_threshold_curves['context'] = ctx
            block_threshold_curves['block_num'] = block_num

            place_fields.append(block_pfs)
            null_curves.append(block_null_curves)
            threshold_curves.append(block_threshold_curves)

        place_fields = pd.concat(place_fields).set_index(
            ['context', 'block_num'], append=True)
        place_fields = place_fields.swaplevel(1, 3).sort_index()

        null_curves = pd.concat(null_curves).set_index(
            ['context', 'block_num'], append=True)
        null_curves = null_curves.swaplevel(1, 3).sort_index()

        threshold_curves = pd.concat(threshold_curves).set_index(
            ['context', 'block_num'], append=True)
        threshold_curves = threshold_curves.sort_index()

        # save results
        with self.place_fields_file(mode='a') as pf_file:
            pf_file.put("place_fields", place_fields)
            pf_file.put("null_curves", null_curves)
            pf_file.put("threshold_curves", threshold_curves)
            print(pf_file.info())

        return place_fields, null_curves, threshold_curves

    def velocity_map(self, tuning_kws={}, **kwargs):
        velo = self.velocity(**kwargs)[self.valid_samples(**kwargs)]
        _, pos, laps = self.get_tuning_inputs(**tuning_kws)

        velo_map = SimpleSpatialTuning().calculate_rasters(
            pd.DataFrame(velo).T, pos, laps, fill_nans=False)

        return velo_map


class PlaceCellContextExperiment(
        ContextMixin,
        SpatialMixin,
        VirtualImagingExperiment
):

    """
    Class for organizing place-cell specific analyses for context switch
    experiments.
    """

    # def analyze_crosscorrelations(self, skip_last_block=True):
    #     """
    #
    #     Parameters
    #     ----------
    #     skip_last_block : bool, optional
    #         True by default. Don't include the last block of familiar context
    #         data at the end of the session.
    #     Returns
    #     -------
    #     results : pd.DataFrame
    #     """
    #
    #     # TODO cache these results in the sima folder so we don't need to
    #     # keep recalculating...
    #
    #     from ca1_novelty.xcorr import compute_block_xcorr, center_of_mass
    #
    #     rasters = self.rasters()
    #     rasters = np.stack([r for _, r in rasters.groupby('roi_label')])
    #     rasters = np.swapaxes(rasters, 0, 1)
    #
    #     results = []
    #     for block_num, (context, (start, stop)) \
    #             in enumerate(self.iter_blocks()):
    #
    #         # by default, only look at the first 4 blocks, since most mice
    #         # didn't run well at the end of the session
    #         if (block_num == 4) & skip_last_block:
    #             continue
    #
    #         xcorr = compute_block_xcorr(rasters[start:stop])
    #         results.append({
    #             'block': block_num,
    #             'context': context,
    #             'xcorr': xcorr,
    #             'com': center_of_mass(xcorr)
    #         })
    #
    #     results = pd.DataFrame(results)
    #     results['mouse_name'] = self.mouse_name
    #     results['expt'] = self.trial_id
    #     results['day'] = self.day
    #
    #     return results

    def analyze_spatial_drift(self, overwrite=False, **kwargs):
        """doc string"""
        # technically this isn't a place cell analysis

        save_path = os.path.join(self.sima_path, "spatial_drift.h5")
        if not overwrite:
            try:
                df = pd.read_hdf(save_path)
                df['mouse'] = self.mouse_name
                return df
            except FileNotFoundError:
                pass

        from ca1_novelty.xcorr import analyze_spatial_drift
        df = analyze_spatial_drift(self, **kwargs)
        df.to_hdf(save_path, "data")
        return df


class ContextSwitchExperimentGroup(ExperimentGroup):

    # use this to organize 3-day PC experiments

    pass


class ContextSwitchDataset(Cohort):

    @property
    def save_path(self):
        from ca1_novelty import __file__ as base_path
        return os.path.join(
            os.path.split(os.path.split(base_path)[0])[0],
            "cache"
        )

    @property
    def field_info_path(self):
        return os.path.join(self.save_path, "nmf_field_info.h5")

    def nmf_field_info(self, recompile=False):
        """Gather all the place field data and metadata for the NMF analysis"""

        if not recompile:
            try:
                return pd.read_hdf(self.field_info_path, "data")
            except Exception:
                print("No field info found. Recompiling dataset...")

        block_sizes = [40, 30, 30, 30]

        results = []
        for expt in self.to_list():

            print(expt.trial_id)

            # get place fields and velocity info
            pfs_df = expt.detect_block_place_fields()[0]
            rasters = expt.block_rasters()
            velocity_map = expt.velocity_map().loc[0]

            expt_results = []
            for roi_label, r in rasters.groupby('roi_label'):
                try:
                    for (block_num, ctx, field_count), pf in \
                            pfs_df.loc[roi_label].iterrows():

                        place_field = PlaceField(
                            pf.start_bin, pf.stop_bin,
                            r.loc[(roi_label, block_num)].values)

                        if place_field.first_lap is None:
                            continue

                        if block_num > 3:
                            continue

                        field_stats = place_field.nmf_field_info()
                        field_stats['roi_label'] = roi_label
                        field_stats['block_num'] = block_num
                        field_stats['context'] = ctx
                        field_stats['field_count'] = field_count
                        field_stats['plateau_velo'] = np.nanmean(
                            velocity_map.loc[
                                field_stats.first_lap
                                + np.sum(block_sizes[0:block_num]),
                                place_field.start_bin:place_field.stop_bin]
                        )

                        expt_results.append(field_stats)

                except KeyError:
                    pass

            expt_results = pd.DataFrame(expt_results)
            expt_results['expt'] = expt.trial_id
            expt_results['day'] = expt.day
            expt_results['set'] = f"{expt.mouse_name}_{expt.set_id}"

            results.append(expt_results)

        results = pd.concat(results)

        # save to disk for fast reloading
        results.to_hdf(self.field_info_path, "data")

        return results

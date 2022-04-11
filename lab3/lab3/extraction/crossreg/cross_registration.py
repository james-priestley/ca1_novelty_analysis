import os
import pickle as pkl

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import seaborn as sns
from IPython.display import clear_output

from sima import ImagingDataset
from sima.ROI import ROIList
from lab3.extraction.crossreg import utils
from lab3.signal import SignalFile


class CrossRegistrar:

    """Class for running ROI cross registration across multiple imaging
    datasets.

    Examples
    --------
    >>> # Create a new CrossRegistrar object
    >>> datasets = [ImagingDataset.load(f) for f in sima_dirs]
    >>> cr = CrossRegistrar(datasets, "rois", savedir="/path/to/folder.xreg")
    >>>
    >>> # Run registration with default options
    >>> cr.register_rois()
    >>>
    >>> # Update signal files of original datasets
    >>> cr.update_signals()
    >>>
    >>> # Load an existing CrossRegistrar object
    >>> cr = CrossRegistrar.load("/path/to/existing/folder.xreg")
    >>> cr.register_rois(overwrite=True)
    >>> cr.update_signals(overwrite=True)

    Parameters
    ----------
    datasets : list
        List of sima ImagingDatasets from which to cross register ROIs
    label : str
        Label of ROI lists to cross register
    savedir : str, optional
        Where to save registration progress. Useful if you want to stop and
        resume processing. Defaults to None, which does not save intermediate
        results.

    Attributes
    ----------
    """

    def __init__(self, datasets, label, savedir=None):

        if datasets is None:
            # Special case used to load an existing cross registration object
            if not savedir:
                raise Exception('Cannot initialize without datasets or a'
                                ' directory.')

            with open(os.path.join(savedir, 'info.pkl'), 'rb') as f:
                info = pkl.load(f)
            self._savedir = savedir
            self.label = info['label']
            self.ds_dirs = info['ds_dirs']
            self.datasets = [ImagingDataset.load(ds_dir)
                             for ds_dir in self.ds_dirs]

        elif all(isinstance(ds, ImagingDataset) for ds in datasets):
            self.savedir = savedir
            self.label = label
            self.datasets = datasets
            self.ds_dirs = [ds.savedir for ds in self.datasets]
            if self.savedir is not None:
                self.save()
        else:
            raise TypeError('CrossRegistrar objects must be initialized with '
                            'a list of sima ImagingDatasets.')

    @property
    def num_datasets(self):
        if not hasattr(self, '_num_datasets'):
            self._num_datasets = len(self.datasets)
        return self._num_datasets

    @property
    def datasets(self):
        return self._datasets

    @datasets.setter
    def datasets(self, datasets):
        num_planes = set([ds.frame_shape[0] for ds in datasets])
        assert len(num_planes) == 1, \
            "All datasets must have the same number of planes."
        assert all(self.label in ds.ROIs.keys() for ds in datasets), \
            f"Not all datasets contain ROI label '{self.label}'"
        self._datasets = datasets
        self.num_planes = list(num_planes)[0]

    @property
    def savedir(self):
        return self._savedir

    @savedir.setter
    def savedir(self, savedir):
        if savedir is None:
            self._savedir = None
        elif hasattr(self, '_savedir') and savedir == self.savedir:
            return
        else:
            if hasattr(self, '_savedir'):
                orig_dir = self.savedir
            else:
                orig_dir = False
            savedir = os.path.abspath(savedir)
            if not savedir.endswith('.xreg'):
                savedir += '.xreg'
            os.makedirs(savedir)
            self._savedir = savedir
            if orig_dir:
                from shutil import copy2
                for f in os.listdir(orig_dir):
                    if f.endswith('.pkl') or f.endswith('.h5'):
                        try:
                            copy2(os.path.join(orig_dir, f), self.savedir)
                        except IOError:
                            pass

    def save(self, savedir=None):
        """Save the CrossRegistrar object to a file."""

        if savedir is None:
            savedir = self.savedir
        self.savedir = savedir

        with open(os.path.join(savedir, 'info.pkl'), 'wb') as f:
            pkl.dump(self._todict(), f, pkl.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path):
        return cls(None, None, path)

    def _todict(self):
        return {'savedir': os.path.abspath(self.savedir),
                'label': self.label,
                'ds_dirs': self.ds_dirs}

    def __str__(self):
        return '<CrossRegistrar>'

    def __repr__(self):
        return (f'<CrossRegistrar: num_datasets={self.num_datasets}, '
                + f'num_planes={self.num_planes}>')

    # ------------------------- Preprocessing steps ------------------------- #

    @property
    def preprocessing(self):
        try:
            return self._preprocessing
        except AttributeError:
            self._preprocessing = pd.read_hdf(
                os.path.join(self.savedir, "preprocessing.h5"))
            return self._preprocessing

    @preprocessing.setter
    def preprocessing(self, df):
        self._preprocessing = df
        if self.savedir is not None:
            df.to_hdf(os.path.join(self.savedir, "preprocessing.h5"),
                      key="data", mode="w")

    @property
    def time_averages(self):
        return self.preprocessing['time_averages']

    @property
    def crops(self):
        return self.preprocessing['crops']

    @property
    def transforms(self):
        return self.preprocessing['transforms']

    @property
    def num_rois(self):
        return self.preprocessing['num_rois']

    @property
    def reference_idx(self):
        return np.where(self.preprocessing['order'].values == 0)[0][0]

    def preprocess(self, suite2p=True, crop=True, overwrite=False):
        """Run necessary preprocessing steps to prepare datasets for
        cross-registration: identify the reference session, find and crop
        the video time averages, and compute transformations to the reference
        image coordinates for later use.

        Parameters
        ----------
        suite2p : bool, optional
            If Suite2p files exist, use the enhanced time average image from
            the ops dictionaries for each plane. Defaults to True
        crop : bool, optional
            Crop image to remove rows or columns with only constant values.
            Recommended for Suite2p processed data. Defaults to True.
        overwrite : bool, optional
            Whether to run preprocessing again if existing data already
            exists.
        """

        if not overwrite:
            try:
                self.preprocessing
                print("Object has already be preprocessed. Set overwrite=True"
                      + " to reprocess")
                return
            except Exception:
                pass

        # initialize dataframe to hold preprocessing results
        df = pd.DataFrame(index=self.ds_dirs,
                          columns=['num_rois', 'order', 'time_averages',
                                   'crops', 'transforms'])

        print("Preparing preprocessing info for registration")
        for ds_dir, df_row in df.iterrows():
            # get number of rois, time averages, and crops for each session
            ds = ImagingDataset.load(ds_dir)
            df_row['num_rois'] = len(ds.ROIs[self.label])

            if suite2p:
                try:
                    time_averages = utils.get_suite2p_time_average(ds)
                except Exception:
                    time_averages = ds.time_averages
            else:
                time_averages = ds.time_averages

            if crop:
                crops = [utils.find_crop_coords(plane)
                         for plane in time_averages]
                df_row['crops'] = crops
                time_averages = np.stack([image[rows][:, columns]
                                          for (rows, columns), image
                                          in zip(*[crops, time_averages])])
            df_row['time_averages'] = time_averages

        # determine processing order by number of ROIs per session
        df['order'] = np.argsort(np.argsort(df['num_rois'] * -1))
        reference_idx = np.where(df.order.values == 0)[0][0]
        self._reference_idx = reference_idx
        print(f"{df.iloc[reference_idx].name} has "
              + f"{df.iloc[reference_idx].num_rois} rois and will serve as "
              + "the reference session")

        # compute transforms to and from the reference session (order 0)
        transforms = []
        for plane in range(self.num_planes):
            print("Calculating transforms to reference session")
            transforms.append(utils.compute_transforms(
                [img[plane] for img in df.time_averages],
                reference_idx=reference_idx))
        df['transforms'] = list(zip(*transforms))

        self.preprocessing = df
        return self.preprocessing

    # ----------------------- Registration functions ------------------------ #

    def _get_dataset(self, order_idx):
        return ImagingDataset.load(self.preprocessing[
            self.preprocessing.order == order_idx].iloc[0].name)

    @property
    def reference_data(self):
        try:
            return self._reference_data
        except AttributeError:
            self._reference_data = pd.read_hdf(
                os.path.join(self.savedir, "reference.h5"))

    @reference_data.setter
    def reference_data(self, df):
        self._reference_data = df
        if self.savedir is not None:
            df.to_hdf(os.path.join(self.savedir, "reference.h5"),
                      key="data", mode="w")

    def _create_reference(self, force_reference=None, overwrite=False):
        """Initialize the reference dataframe

        Parameters
        ----------
        force_reference : str, optional
            Path to sima folder of reference dataset. If None (default),
            the dataset with the greatest number of cells is used as the
            reference.
        overwrite : bool, optional
            Recompile the reference from scratch
        """
        if not overwrite:
            try:
                self.reference_data
                print("Using existing reference data. Set overwrite=True to"
                      + " recompile")
                return
            except Exception:
                pass

        # find the reference dataset
        if force_reference:
            ref_ds = ImagingDataset.load(force_reference)
        else:
            ref_ds = self._get_dataset(0)
        crop = self.crops.loc[ref_ds.savedir]

        # get dataframe of ROI information and save as attribute
        reference_data = pd.DataFrame(
            [utils.get_roi_info(roi, crop=crop)
             for roi in ref_ds.ROIs[self.label]])
        reference_data['source_dataset'] = ref_ds.savedir
        reference_data['partners'] = \
            [{roi.source_dataset: label} for label, roi
             in reference_data.iterrows()]
        reference_data['mask_pairs'] = \
            [{roi.source_dataset: roi['mask']} for label, roi
             in reference_data.iterrows()]
        self.reference_data = reference_data
        return self.reference_data

    @property
    def _reference_size(self):
        return self.preprocessing.loc[
            self.reference_data.iloc[0].source_dataset].time_averages.shape

    def _pair_dataset(self, idx, pairing_threshold=0.5, max_distance=25):
        """Do pairing for dataset against the current reference stack

        Parameters
        ----------
        pairing_threshold : float, optional
        max_distance : int, optional
        """

        ds = self._get_dataset(idx)
        if ds.savedir in self.reference_data.source_dataset.values:
            print(f"'{ds.savedir}' is already paired, or was the reference "
                  + "session. Skipping...")
            return

        print(f"Pairing {self.num_rois[ds.savedir]} ROIs from {ds.savedir}")

        print(" -- Transforming ROIs to reference")
        rois = pd.DataFrame(
            [utils.get_roi_info(roi, crop=self.crops[ds.savedir],
                                transform=self.transforms[ds.savedir],
                                target_shape=self._reference_size)
             for roi in ds.ROIs[self.label]])

        reference_data = self.reference_data

        for plane in np.unique(rois.plane):
            print(f" -- Plane {plane}")

            plane_rois = rois[rois.plane == plane]
            reference_rois = reference_data[reference_data.plane == plane]
            print(f" ---- Pairing {len(plane_rois)} ROIs to"
                  + f" {len(reference_rois)} references")

            print(" ---- Computing distance matrix")
            dist_mat = utils.calc_distance_matrix(reference_rois['mask'],
                                                  plane_rois['mask'])

            # binarize and find optimal pairs via Hungarian algorithm
            print(" ---- Finding optimal pairs")
            dist_mat[dist_mat > pairing_threshold] = 1  # binarize at threshold
            roi_pairs = linear_sum_assignment(dist_mat)
            roi_pairs = [(reference_rois.iloc[x].name, plane_rois.iloc[y].name)
                         for x, y in zip(*roi_pairs)
                         if dist_mat[x, y] < pairing_threshold]

            print(f" ---- Autopaired {len(roi_pairs)}/{len(plane_rois)} ROIs")
            for ref_label, new_label in roi_pairs:
                reference_data.loc[ref_label].partners[ds.savedir] = new_label
                plane_rois.drop(index=new_label, inplace=True)

            print(f" ---- Starting manual review of remaining ROIs")
            total_unpaired = len(plane_rois)
            for uidx, (label, unpaired_roi) in \
                    enumerate(plane_rois.iterrows()):
                print(f"Reviewing ROI {uidx + 1}/{total_unpaired}")

                eligible_ref_rois = utils.find_nearby_rois(
                    unpaired_roi.centroid, reference_rois,
                    max_distance=max_distance, exclude_partnered=ds.savedir)

                if eligible_ref_rois.empty:
                    print(f"No nearby ROIs, adding as new entry to reference")
                else:
                    ref_label = self._plot_manual_pair(
                        unpaired_roi, eligible_ref_rois,
                        max_distance=max_distance)

                    if ref_label is not None:
                        if ref_label == -1:
                            print("Rejected ROI")
                        else:
                            reference_data.loc[ref_label].partners[
                                ds.savedir] = unpaired_roi.name
                            reference_data.loc[ref_label].mask_pairs[
                                ds.savedir] = unpaired_roi['mask']
                        plane_rois.drop(index=unpaired_roi.name, inplace=True)
                        continue
                    else:
                        print("No matching ROIs, adding as new entry to "
                              + "reference")

                # add unpaired ROI as a new entry to reference if a pair was
                # not identified
                unpaired_roi['source_dataset'] = ds.savedir
                unpaired_roi['partners'] = {ds.savedir: unpaired_roi.name}
                unpaired_roi['mask_pairs'] = {ds.savedir: unpaired_roi['mask']}
                reference_data.loc[unpaired_roi.name] = unpaired_roi
                plane_rois.drop(index=unpaired_roi.name, inplace=True)

        # resave reference
        print("ROIs registered for all planes.\n"
              + f"Total count is {len(reference_data)}.\n"
              + "ROIs appear in {} ".format(
                np.mean([len(p) for p in reference_data.partners]))
              + "sessions on average.\n"
              + f"Updating reference for future registrations")
        self.reference_data = reference_data
        return self.reference_data

    def _is_in_bounds(self, roi):
        mask = roi['mask']
        for source_dataset, session in self.preprocessing.iterrows():
            if source_dataset == roi.source_dataset:
                continue

            new_mask = utils.apply_transform(
                mask, session.transforms[roi.plane],
                session.time_averages.shape[1:], invert=True)

            if new_mask.sum() == 0:
                return False
        return True

    def remove_out_of_bound_rois(self):
        reference_data = self.reference_data
        print("Checking list of {} ROIs for out-of-bounds masks".format(
            len(reference_data)))
        for label, roi_info in reference_data.iterrows():
            if not self._is_in_bounds(roi_info):
                print(f"...removed {label}")
                reference_data.drop(index=label, inplace=True)
        self.reference_data = reference_data
        print(f"{len(reference_data)} ROIs remain")
        return self.reference_data

    def register_rois(self, pairing_threshold=0.5, max_distance=25,
                      remove_out_of_bound_rois=True,
                      force_reference=None, overwrite=False,
                      preprocessing_kws={}):
        """Main registration function. Iterate over ImagingDatasets and
        pair ROIs with the reference dataset. As new ROIs are found, they are
        added to the reference. The augmented reference set is then used to
        pair the next dataset, and so the reference grows over time and
        facilitates additional autopairing.

        Parameters
        ----------
        pairing_threshold : float [0, 1], optional
            Jaccard score threshold for autopairing. ROI pairs with scores
            below (?) this threshold will be flagged for manual review.
            Defaults to 0.5
        max_distance : float, optional
            Maximum x and y distance in pixels between ROI centroids for
            considering candidate reference ROIs to pair under manual review
        remove_out_of_bound_rois : bool, optional
            After pairing, remove any ROIs that would fall outside of the FOV
            in any single session. We apply the inverse transform (from
            the reference to each session) on each ROI in the final list and
            make sure the masks are non-empty in every session). Defaults to
            True.
        force_reference : str, optional
            ImagingDataset.savedir of dataset to force as the reference.
            Defaults to None, in which case the dataset with the greatest
            number of ROIs is chosen.
        overwrite : bool, optional
            If False, registration will resume from previous work. If True,
            prior results will be cleared and registration will start from
            scratch.
        preprocessing_kws : dict
            Pass additional arguments to the preprocessing pipeline.
        """

        self.preprocess(overwrite=overwrite, **preprocessing_kws)
        self._create_reference(force_reference=force_reference,
                               overwrite=overwrite)

        for idx in range(len(self.ds_dirs)):
            self._pair_dataset(idx, pairing_threshold=pairing_threshold,
                               max_distance=max_distance)

        if remove_out_of_bound_rois:
            self.remove_out_of_bound_rois()

    # ------------------------- Plotting functions -------------------------- #

    def _plot_manual_pair(self, unpaired_roi, eligible_ref_rois,
                          max_distance=25):
        """Generates an interactive plot to choose the correct ROI"""

        time_avg = self.time_averages[
            eligible_ref_rois.iloc[0].source_dataset][unpaired_roi.plane]
        ylims, xlims = [[-max_distance, max_distance] + c
                        for c in unpaired_roi.centroid]

        plt.figure(figsize=(6, 6))
        sns.heatmap(time_avg, square=True, cbar=False, cmap='Greys_r')

        mask_stack = np.stack(eligible_ref_rois['mask']).sum(axis=0)
        sns.heatmap(mask_stack, mask=(mask_stack == 0), alpha=0.4, cmap='Reds',
                    cbar=False)
        for num, centroid in enumerate(eligible_ref_rois.centroid):
            text = plt.text(centroid[1], centroid[0], num, fontsize=25,
                            horizontalalignment='center',
                            verticalalignment='center', color='red',
                            weight='bold')
            text.set_path_effects([path_effects.Stroke(linewidth=1,
                                                       foreground='black'),
                                   path_effects.Normal()])

        sns.heatmap(unpaired_roi['mask'],
                    mask=~unpaired_roi['mask'].astype('bool'),
                    vmin=0, vmax=1, alpha=0.4, cmap='Blues', cbar=False)

        plt.title('Available ROIs')
        plt.xlim(xlims)
        plt.ylim(ylims)
        plt.axis('off')
        plt.show()

        # let the user pick which cell to retain
        while True:
            decision = input(
                "Choose the matching ROI\n(Press enter if new, -1 to discard)")
            try:
                if decision == '':
                    clear_output()
                    return None
                elif (int(decision) in np.arange(len(eligible_ref_rois))):
                    clear_output()
                    return eligible_ref_rois.iloc[int(decision)].name
                elif int(decision) == -1:
                    clear_output()
                    return -1
                else:
                    print('Invalid input!')
            except ValueError:
                print("Invalid input!")

    def _plot_avg_mask(self, roi):
        # TODO
        pass

    def _plot_cumulative_fov(self):
        # TODO
        pass

    # ---------------------- Signal splitting functions --------------------- #

    def update_signals(self, overwrite=False):
        """Update signals dataframes for the underlying datasets. A new
        ROI list is stored that contains all ROIs (whether they appear in
        a given session or not). If an ROI is not present in a particular
        session, its trae appears in the signal dataframes as NaN.

        Parameters
        ----------
        overwrite : bool, optional
            Whether to overwrite the registered ROIs, if they already exist.
            Defaults to True.
        """
        for sima_dir in self.preprocessing.index:
            print(f"Saving results for {sima_dir}")

            ds = ImagingDataset.load(sima_dir)
            original_rois = ds.ROIs[self.label]
            if len(original_rois) == 0:
                print(" - Session has no ROIs, skipping...")
                continue

            with SignalFile(os.path.join(sima_dir, "signals.h5")) \
                    as signal_file:

                # find all signals dataframes for self.label
                keys_to_update = [key for key in signal_file.keys()
                                  if f'/{self.label}/' in key]
                for key in keys_to_update:
                    new_key = key.replace(f'/{self.label}/',
                                          f'/{self.label}_registered/')
                    print(f" - Updating {key} to {new_key}")
                    if new_key in signal_file:
                        if not overwrite:
                            print(f"Signals already exist for {new_key}. Set "
                                  + "overwrite=True to re-import. Skipping...")
                            continue
                        print(f" - Deleting previous import at {new_key}")
                        signal_file.remove(new_key)

                    # compile new dataframe with all ROIs
                    df = signal_file.get(key)
                    new_df = pd.DataFrame(index=[], columns=df.columns)
                    new_df.index.name = df.index.name
                    for new_label, roi_info in self.reference_data.iterrows():
                        try:
                            # find the name of the ROI in the current session
                            new_df.loc[new_label] = \
                                df.loc[roi_info.partners[sima_dir]]
                        except KeyError:
                            new_df.loc[new_label] = np.nan

                    # save
                    signal_file.put(new_key, new_df)

            # add ROI list to the dataset
            # TODO - should we correct the masks to be of the right dimensions
            #        for the given dataset?
            print(" - Adding ROIList to dataset")
            ds.add_ROIs(ROIList(self.reference_data.roi.values),
                        f'{self.label}_registered')

    # -------------------------- Cleanup functions -------------------------- #

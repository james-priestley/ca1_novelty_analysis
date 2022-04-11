"""Classes for concatenating and batch-processing multiple recordings via
Suite2p and SIMA"""

import os
import pickle
import sima

from . import Suite2pImagingDataset
from ...signal import SignalFile


class BatchMixin:

    """Mixin class that modifies the functionality of sima ImagingDatasets.
    This facilitates the construction of multi-sequence datasets from existing
    sima datasets, for batch processing. The directories of the original child
    datasets are retained, so that results can be later applied to those
    datasets. Datasets derived from BatchMixin can be initialized in several
    ways, depending on the passed parameters.

    Parameters
    ----------
    sequences : list of Sequence objects, optional
        Sequences to include in the batch dataset. If passed, you must also
        pass 'child_dirs' (see below).
    child_dirs : list of str, optional
        A list of sima directories. If 'sequences' is passed, this is a list of
        the same length; a sima ImagingDataset will be created for each
        sequence and saved in these directories. If 'child_dirs' is passed
        alone, the batch dataset will be created from existing sima datasets
        at these locations (if they exist).
    ds_list : list of ImagingDataset objects, optional
        The batch dataset is created by combining sequences from the passed
        ImagingDataset objects.
    savedir : str, optional
        Where to save the batch dataset
    channel_names : list of str, optional
        If not passed, the channel names are taken from the first child
        dataset in the batch.

    Attributes
    ----------
    child_dirs : list of str
        A list of sima directories, corresponding to each of the child
        datasets
    """

    def __init__(self, sequences=None, savedir=None, child_dirs=None,
                 ds_list=None, channel_names=None):

        # TODO Is there a more pythonic way of doing this?
        try:
            self.base_class = self.__class__.__bases__[1]
        except Exception:
            self.base_class = sima.ImagingDataset

        if (not sequences) and (not child_dirs) and (not ds_list):
            # Load an existing batch dataset
            super().__init__(None, savedir)

            with open(os.path.join(savedir, 'dataset.pkl'), 'rb') as f:
                data = pickle.load(f)

            # Add child directory information as an attribute
            try:
                self.child_dirs = data.pop('child_dirs')
            except KeyError:
                raise Exception(f'{savedir} is not a batched dataset')
            return

        elif sequences:
            # Create a new batch dataset from sequences, and create child sima
            # folders
            if (child_dirs is None) or (len(child_dirs) != len(sequences)):
                raise Exception("A child directory must be specified for "
                                + "every sequence")
            elif ds_list:
                raise ValueError("Either 'sequences' or 'ds_list' must be "
                                 + "specified, but not both")
            else:
                # create child sima directories
                if any([os.path.exists(child_dir)
                        for child_dir in child_dirs]):
                    raise Exception("At least one child directory already "
                                    + "exists")
                ds_list = [self.base_class(sequence, child_dir,
                                           channel_names=channel_names)
                           for sequence, child_dir in
                           zip(*[sequences, child_dirs])]

        elif child_dirs:
            # Create a new batch dataset from existing sima directories
            if ds_list:
                raise Exception('TODO')
            else:
                ds_list = [self.base_class.load(child_dir)
                           for child_dir in child_dirs]

        if ds_list:
            # Create batch dataset
            if any([len(ds.sequences) > 1 for ds in ds_list]):
                raise Exception("Each child dataset can only have one "
                                + "sequence")
            elif len(set([ds.frame_shape for ds in ds_list])) > 1:
                raise Exception("All datasets must have compatible frame "
                                + "shapes")
            else:
                if channel_names is None:
                    channel_names = ds_list[0].channel_names
                sequences = [ds.sequences[0] for ds in ds_list]
                self.child_dirs = [ds.savedir for ds in ds_list]
                super().__init__(sequences, savedir,
                                 channel_names=channel_names)
        else:
            raise TypeError("New batch datasets must be initialized from "
                            + "'sequences'/'child_dirs', 'child_dirs', or "
                            + "'ds_list'.")

    def _todict(self):
        return {**super()._todict(), **{'child_dirs': self.child_dirs}}

    @property
    def children(self):
        child_datasets = []
        for path in self.child_dirs:
            child_datasets.append(self.base_class.load(path))
        return child_datasets


class BatchSuite2pImagingDataset(BatchMixin, Suite2pImagingDataset):

    """Class for running batch processing of recordings via Suite2p. Datasets
    are concatenated and jointly motion corrected, if desired. ROI detection
    and signal extraction are performed on the concatenated dataset. Following
    curation of ROIs in the Suite2p GUI, the class provides a method to
    split the signals and farm them back to the original child datasets.

    Parameters
    ----------
    sequences : list of Sequence objects, optional
        Sequences to include in the batch dataset. If passed, you must also
        pass 'child_dirs' (see below).
    child_dirs : list of str, optional
        A list of sima directories. If 'sequences' is passed, this is a list of
        the same length; a sima ImagingDataset will be created for each
        sequence and saved in these directories. If 'child_dirs' is passed
        alone, the batch dataset will be created from existing sima datasets
        at these locations (if they exist).
    ds_list : list of ImagingDataset objects, optional
        The batch dataset is created by combining sequences from the passed
        ImagingDataset objects.
    savedir : str, optional
        Where to save the batch dataset
    channel_names : list of str, optional
        If not passed, the channel names are taken from the first child
        dataset in the batch.

    Attributes
    ----------
    child_dirs : list of str
        A list of sima directories, corresponding to each of the child
        datasets
    """

    def import_results_to_children(self, label='suite2p', channel='Ch2',
                                   overwrite=False):
        """Split concatenated Suite2p results amongst the child datasets
        associated with this batched dataset.

        Parameters
        ----------
        label : str, optional
            ROI label to store the signals and ROI masks under. Defaults to
            'suite2p'.
        channel : {str or int}, optional
            Name or index of dynamic channel in the imaging dataset. Defaults
            to 'Ch2'.
        overwrite : bool, optional
            If key '/{channel}/{label}' already exists in the dataset signals
            file, overwrite with new import. Note this will delete any
            additional signal types in the store in addition to the 'raw' and
            'npil' (e.g. 'dfof'), to prevent mixtures of signals derived from
            different imports. Defaults to False, which will raise an error if
            the desired key already exists.
        """

        # Maybe functionalize repeated elements from the standard import
        # method

        # if the concatenated results have not been imported, do so
        if not os.path.exists(os.path.join(self.savedir, "signals.h5")):
            print("No parent signals found. Importing parent dataset first:")
            self.import_results(label=label, channel=channel,
                                overwrite=overwrite)

        # find all signals records for matching the desired channel/label
        with SignalFile(os.path.join(self.savedir, "signals.h5")) \
                as parent_signal_file:

            base_key = f"/Ch2/{label}"
            keys_to_split = [key for key in parent_signal_file.keys()
                             if base_key in key]
            data_to_split = {key: parent_signal_file[key]
                             for key in keys_to_split}

            if len(keys_to_split):
                print(f"Signal keys to split among children: {keys_to_split}")
            else:
                raise KeyError(f"No parent signals found under {base_key}")

        # slice concatenated signals and save under each child sima dataset
        frame_count = 0
        for child in self.children:
            child_slice = slice(frame_count, frame_count + child.num_frames)
            print(f"\n{child.savedir} <--> {child_slice}")
            frame_count += child.num_frames

            with SignalFile(os.path.join(child.savedir, "signals.h5")) \
                    as child_signal_file:

                for key, df in data_to_split.items():
                    if key in child_signal_file:
                        assert overwrite, f"Signals already exist for {key}." \
                            + " Set overwrite=True to re-import.\nCurrent " \
                            + f"file structure:\n {child_signal_file.info()}"

                        print(f"Deleting previous import at {key}")
                        child_signal_file.remove(key)

                    # add signals to child dataset
                    child_signal_file.put(
                        key,
                        df.iloc[:, child_slice].T.reset_index(drop=True).T)

                print(child_signal_file.info())

            # add ROIs to child dataset (TODO: another overwrite check?)
            child.add_ROIs(self.ROIs[label], label)

    def apply_mc_to_children(self, overwrite=True):
        """doc string"""

        frame_count = 0
        for child in self.children:
            child_slice = slice(frame_count, frame_count + child.num_frames)
            print(f"\n Applying mc to {child.savedir} <--> {child_slice}")

            frame_count += child.num_frames
            child_ops = self.ops_list
            for plane in range(len(child_ops)):
                child_ops[plane]['xoff'] = \
                    child_ops[plane]['xoff'][child_slice]
                child_ops[plane]['yoff'] = \
                    child_ops[plane]['yoff'][child_slice]

            child.apply_mc_results(overwrite=overwrite, ops_list=child_ops)

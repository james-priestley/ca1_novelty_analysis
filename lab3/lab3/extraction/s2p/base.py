"""Dedicated ImagingDataset classes for extracting and importing signals via
Suite2p"""

import os
import shutil
import warnings

# import before numpy so we set the environment variables
from lab3.extraction.s2p import settings, extract_signals, import_signals
from lab3.misc.sima_sequences import _Suite2pRigidSequence
# from lab3.misc.sima_helpers import backup_dir, get_base_sequence
from lab3.misc.sima_compatibility import sima_compatible

import numpy as np
from abc import abstractmethod, ABCMeta
from sima import ImagingDataset
from pkg_resources import get_distribution


class Suite2pStrategy(ImagingDataset, metaclass=ABCMeta):

    """Abstract class for creating Suite2p strategies. We inherit from
    sima.ImagingDataset, in order to interface with the underlying
    imaging data. Accordingly, instances should be generated via the load
    method and not instantiated directly (i.e. we assume a sima folder
    already exists).

    Implement :meth:`__init__`, :meth:`extract`, and :meth:`import_results` to
    subclass.

    See also
    --------
    sima.ImagingDataset
    """

    def __str__(self):
        return f'<{type(self).__name__}>'

    def __repr__(self):
        return ('<Suite2pImagingDataset: '
                + f'num_sequences={self.num_sequences},'
                + f'frame_shape={self.frame_shape}, '
                + f'num_frames={self.num_frames}>')

    @property
    def s2p_path(self):
        return os.path.join(self.savedir, "suite2p")

    @property
    def ops_list_path(self):
        return os.path.join(self.s2p_path, "ops1.npy")

    @property
    def ops(self):
        return self._ops

    @ops.setter
    def ops(self, user_ops):
        """Creates the ops dictionary for Suite2p input

        Parameters
        ----------
        user_ops : dict
            Pass keyword/argument pairs as a dictionary to override
            the default Suite2p settings. See OPS for option details.
        """

        user_ops["xrange"] = np.array([0, self.frame_shape[2]])
        user_ops["yrange"] = np.array([0, self.frame_shape[1]])
        self._ops = {**settings.get_ops(
            get_distribution("suite2p").version),
                     **user_ops}

    @property
    def ops_list(self):
        try:
            return np.load(self.ops_list_path, allow_pickle=True)
        except FileNotFoundError:
            raise AttributeError("ops_list does not exist. Call extract to "
                                 + "process this dataset first.")

    @property
    def db(self):
        """Creates the db dictionary for Suite2p input"""
        # merge with default db
        # TODO: There should be a way to get this to work, looks like
        # there are some 'under the hood' issues with shutil
        # 'fast_disk': os.path.join("/fastscratch/scratch/suite2p/fast_disk",
        #    os.path.basename(self.savedir))}}
        return {**settings.DB, **{'data_path': [self.savedir],
                                  'save_path0': self.savedir,
                                  'fast_disk': self.savedir}}

    @property
    def bad_frames_path(self):
        return os.path.join(self.savedir, "bad_frames.npy")

    @property
    def bad_frames(self):
        """Frames to ignore during processing"""
        return np.load(self.bad_frames_path)

    @property
    def results_path(self):
        """For multiplane data assume the user has curated the combined view"""
        subfolder = "combined" if os.path.exists(
            os.path.join(self.s2p_path, "combined")) else "plane0"
        return os.path.join(self.s2p_path, subfolder)

    @property
    def raw_signals(self):
        """Returns the raw fluorescence traces for each ROI"""
        return np.load(os.path.join(self.results_path, "F.npy"),
                       allow_pickle=True)

    @property
    def npil_signals(self):
        """Returns the neuropil fluorescence traces for each ROI"""
        return np.load(os.path.join(self.results_path, "Fneu.npy"),
                       allow_pickle=True)

    @property
    def cell_indicator(self):
        """Returns a list of row indices corresponding to accepted cells"""
        try:
            return self._cell_indicator
        except AttributeError:
            self._cell_indicator = np.where(
                np.load(os.path.join(self.results_path, "iscell.npy"),
                        allow_pickle=True)[:, 0])[0]
            return self._cell_indicator

    def _modify_cell_indicator(self, value):
        assert value in [0, 1], "value must be 0 (False) or 1 (True)!"
        try:
            del self._cell_indicator
        except AttributeError:
            pass
        ci = np.load(os.path.join(self.results_path, "iscell.npy"),
                     allow_pickle=True)
        ci[:, 0] = value
        np.save(os.path.join(self.results_path, "iscell.npy"), ci)

    def accept_all_cells(self):
        """Classify all ROIs as cells"""
        self._modify_cell_indicator(1)

    def reject_all_cells(self):
        """Classify all ROIs as not cells"""
        self._modify_cell_indicator(0)

    @property
    def static_indicator(self):
        """Returns a list of row indices corresponding to cells with the
        static marker"""
        raise NotImplementedError

    def get_stat_file(self, plane=0):
        """Returns list of dictionaries with additional ROI information"""
        return np.load(os.path.join(
            self.s2p_path, f"plane{plane}", "stat.npy"), allow_pickle=True)

    # ----------------- Implement these methods to subclass ----------------- #

    @abstractmethod
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @abstractmethod
    def extract(self):
        pass

    @abstractmethod
    def import_results(self):
        pass

    # ------------------------------- Optional ------------------------------ #

    def apply_mc_results(self):
        pass


@sima_compatible
class Suite2pImagingDataset(Suite2pStrategy):

    """The main interface for using Suite2p with SIMA. Suite2pImagingDatasets
    are instantiated via the load method, inherited from ImagingDataset. They
    are not initialized directly.

    Examples
    --------
    >>> from lab3.extraction.s2p import Suite2pImagingDataset
    >>> ds = Suite2pImagingDataset.load("/path/to/my/folder.sima")
    >>> ds.extract()

    Following extraction, the user must manually curate the results in the
    Suite2p GUI on their local machine. Signals can then be imported using
    the object again:

    >>> ds.import_results()
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def extract(self, overwrite=False, reprocess=False, **kwargs):
        """Run Suite2p extraction on imaging dataset.

        Prior to conversion, the imaging dataset is converted to a binary file,
        in the format expected by Suite2p. Along the way, additional 'ops'
        dictionaries are created for each imaging plane and saved to disk in
        the suite2p directory.

        If the dataset has multiple sequences, they will be concatenated in the
        binary file as if they were one continuous recording. For this reason,
        all sequences should be of compatible frame shape (zyxc). This allows
        batch processing of datasets from the same FOV, but special
        consideration should be given for motion correction. If the individual
        datasets have already been motion corrected, you must align the
        sequences and crop them accordingly when constructing the batch imaging
        dataset object. This step is unnecessary if using Suite2p's motion
        correction, as the frames of all sequences will be aligned to a common
        reference.

        Parameters
        ----------
        signal_channel : str or int, optional
            Name or index of the dynamic channel to be extracted.
        static_channel : str or int, optional
            Name or index of the static channel. Provide this if you want
            to for example use Suite2p's red cell detection. To determine
            registration shifts using the static channel, override
            'align_by_chan' via ops_kws (see below), setting it equal to 2.
        fs : float, optional
            Frame rate (per-plane), in Hz. Defaults to 10.
        collapse_z : bool, optional
            Whether to average all planes on each frame prior to processing.
            Defaults to True.
        fill_gaps : bool, optional
            Whether to fill NaNed values with data from adjacent frames.
            Defaults to None, which will fill_gaps only if register is False.
        remove_masked_frames : bool, optional
            Skip masked frames when creating binary file. They will be
            reinserted as NaNed entries at the proper timepoints in the final
            imported signal traces.
            Not implemented.
        register : bool, optional
            Whether to use Suite2p motion correction, which is nonrigid by
            default. To use the rigid motion correction only, override
            'nonrigid' via ops_kws (see below), setting it equal to False.
            By default, no registration is performed.
        sparse_mode : bool, optional
            Whether to use Suite2p's sparse mode. Defaults to True
        spatial_scale : int {0, 1, 2, 3}, optional
            ROI scale parameter for sparse model. 0 = multiscale, 1 = 6 pixels,
            2 = 12 pixels, 3 = 24 pixels, 4 = 48 pixels. Choose the closest
            value. Defaults to 0.
        diameter : int or list of ints, optional
            Expected diameter of cells, in pixels. Defaults to None. Ignored
            if sparse_mode is True. If sparse_mode is False, you must specify
            a diameter. Pass a list of two ints to specify different diameters
            for x and y dimensions.
        overwrite : bool, optional
            If True, existing Suite2p folder is deleted and dataset is
            reprocessed from the beginning of the pipeline. Defaults to False.
        reprocess : bool, optional
            If True, rerun Suite2p analysis on existing binary file.
            This can be used for example to extract ROIs multiple times with
            different parameters, without repeating the binary conversion and
            registration each time. Reprocess and overwrite cannot both be
            True.
            Not tested - we may need to rewrite the ops dictionaries.
        ops_kws : dict, optional
            Pass additional keyword-argument pairs as a dictionary to
            override other Suite2p default settings.

        See also
        --------
        lab3.extraction.s2p_helpers.dump_to_binary
        lab3.extraction.s2p_helpers.extract
        """

        # overwrite checks
        assert (not (overwrite & reprocess)), \
            "overwrite and reprocess cannot both be true"
        if os.path.isdir(self.s2p_path):
            if overwrite:
                print('Removing previous Suite2p folder')
                shutil.rmtree(self.s2p_path)
            elif reprocess:
                warnings.warn("reprocess is untested.")
                pass
            else:
                print("Suite2p path:\n%s\n...already exists, exiting..."
                      % self.s2p_path)
                return

        # run extraction
        extract_signals.extract(self, reprocess=reprocess, **kwargs)

    def import_results(self, **kwargs):
        """Import Suite2p signal results using the new-style signal formats
        (dataframes and h5 stores).

        Parameters
        ----------
        ds : Suite2pStrategy
            Instance of a Suite2pStrategy, which has been previously extracted.
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

        See also
        --------
        lab3.extraction.s2p_helpers.import_to_signals_file
        """

        import_signals.import_to_signals_file(self, **kwargs)

    def apply_mc_results(self, ops_list=None, overwrite=False):
        """Apply displacements calculated by Suite2p to the underlying dataset.
        Currently only supports rigid motion correction.

        Parameters
        ----------
        ds : Suite2pStrategy
            Instance of a Suite2pStrategy, which has been previously extracted.
        ops_list : array of dict, optional
            Array containing suite2p options and MC results for each plane.
            Pass only if you want to use suite2p MC results from somewhere else
            If not specified, will look for suite2p results native to this
            dataset.
        overwrite : bool, optional
            Whether to save the motion corrected dataset as a sima directory.
            Will back up the current sima directory, if it exists.
        """

        if ops_list is None:
            ops_list = self.ops_list

        new_sequences = []

        for i in range(self.num_sequences):
            if ops_list[0]['nonrigid']:
                # TODO implement nonrigid sequence wrapper
                raise NotImplementedError
            else:
                seq = _Suite2pRigidSequence(self.sequences[i], ops_list)
                new_sequences.append(seq)

        self.sequences = new_sequences

        if overwrite:
            # backup_dir(self.savedir, delete_original=False)
            self.save()


class OldSuite2pImagingDataset(Suite2pStrategy):

    """For backwards compatibility, this strategy replicates the core
    functionality of the lab's suite2p autoscript, for Suite2p version 0.5.5.
    Additional options are provided to store the extracted signals in the old
    signals format.

    Examples
    --------

    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def extract(self):
        """Not implemented

        See also
        --------
        lab3.extraction.s2p_helpers.extract_with_055
        """

        extract_signals.extract_with_055()

    def import_results(self, as_pickle=False):
        """Not implemented"""

        if as_pickle:
            import_signals.import_to_pkl()
        else:
            raise NotImplementedError

"""A collection of base classes for representing the basic experiment types
used in the lab: combinations of behavior, imaging, and LFP recordings.
Inherit from these to extend their functionality for more paradigms, e.g.
specific behavior tasks, that lend themselves to bespoke analysis methods."""

import os
import json
import functools
import pickle as pkl
import warnings
import glob
import pandas as pd
import h5py

import numpy as np
from sima.imaging import ImagingDataset

from lab3.core import Item
from lab3.core.metadata import DefaultMetadata
from lab3.experiment import utils, database
from lab3.experiment.decorators import default_label_handling
from lab3.signal.base import SignalFile
from lab3.signal.utils import nan_zscore, nonparametric_nan_zscore

from lab3.event import BehaviorEventGroup, LFPEventGroup, EventGroup
from lab3.event.transient import ExperimentTransientGroup

# Accomodate old naming conventions
KEY_DICT = {
    'startTime': 'start_time',
    'time': 'start_time',
    'stopTime': 'stop_time',
    'tSeriesDirectory': 'tSeries_path',
    'project_name': 'experiment_group'
}


def fetch_trials(*args, **kwargs):
    """Pass keyword arguments specifying attribute and value pairs from
    the database, to get a list of trial IDs satisfying those conditions.
    Some keyword arguments are automatically renamed to match the database
    conventions. Some common attributes are listed below.

    Parameters
    ----------
    args : str, optional
        Name of trial attributes required to be present
    project_name : str, optional
        By convention, this is usually the experimenter's name
    mouse_name : str, optional
    experimentType : str, optional
    start_time : str
    stop_time : str

    Returns
    -------
    trial_ids : list

    Examples
    --------
    >>> trial_ids = fetch_trials(project_name='test', mouse_name='test')
    >>> imaged_trials = fetch_trials('tSeriesDirectory', mouse_name='test')

    See also
    --------
    lab3.experiment.database
    """

    # TODO should we continue to support this, or just strictly adopt the names
    # of the database?

    args = list(args)

    for k in KEY_DICT:
        if k in kwargs:
            kwargs[KEY_DICT[k]] = kwargs.pop(k)

        if k in args:
            args[args.index(k)] = KEY_DICT[k]

    return database.fetch_trials(*args, **kwargs)


class _BaseExperiment(Item):

    """Class establishing some basic experiment attributes. Not intended to be
    used directly"""

    constructor_ids = ['trial_id']

    def __init__(self, trial_id=None, mouse_id=''):
        self.trial_id = trial_id
        self._trial_info = {'trial_id': trial_id, 'mouse_id': mouse_id}

    @property
    def mouse_id(self):
        return self._trial_info['mouse_id']

    @property
    def experiment_type(self):
        try:
            return self._trial_info['experimentType']
        except KeyError:
            return ''

    def events(self, event_type, signal_type='imaging', channel=None,
               label=None, **kwargs):
        if signal_type == 'imaging':
            return self.imaging_events(event_type=event_type, channel=channel,
                                       label=label, **kwargs)
        elif signal_type == 'LFP':
            return self.lfp_events(event_type, channel=channel, label=label,
                                   **kwargs)
        elif signal_type == 'behavior':
            return self.behavior_events(event_type=event_type, **kwargs)

    def signals(self, signal_type='raw', channel='Ch2',
               label=None, max_frame=None, as_time_signals=False, 
               source='imaging', metadata=DefaultMetadata, **kwargs):
        if source == 'behavior':
            signals = self.behavior_signals(
                label=label, channel=channel, as_time_signals=as_time_signals,
                 **kwargs)
        elif source == 'LFP':
            signals = self.lfp_signals(
                signal_type=signal_type, channel=channel, label=label, 
                as_time_signals=as_time_signals, **kwargs)
        elif source == 'imaging':
            signals = self.imaging_signals(
                signal_type=signal_type, channel=channel, 
                as_time_signals=as_time_signals, label=label, **kwargs)
        else:
            signals = source(self, **kwargs)

        try:
            signals = metadata.bind_to(signals, expt=self)
        except Exception as exc:
            print(f"Binding metadata : {metadata} to {self} failed with {exc}")

        if max_frame is not None:
            return signals.iloc[:, :max_frame]
        else:
            return signals

class BehaviorExperiment(_BaseExperiment):

    """Base class for all experiments in the behavior database.

    Parameters
    ----------
    trial_id : int
        ID of experiment in the sql database
    """

    def __init__(self, trial_id):
        super().__init__(trial_id=trial_id)

        # get experiment attributes from the database
        # NOTE : We've dramatically simplified the loading of experiment
        # attributes from the database. There might be some growing pains for
        # things that are no longer automatically exposed that will need to be
        # reimplemented

        try:
            self._trial_info.update({
                **database.fetch_trial(trial_id),
                **database.fetch_all_trial_attrs(trial_id, parse=True)
            })
            self._trial_info_clean = self._trial_info.copy()
        except TypeError:
            raise KeyError(f"Trial ID {trial_id} does not exist")

    def identifiers(self):
        return {
            'trial_id': self.trial_id,
            'mouse_id': self.mouse_name,
            'expt_type': self.experiment_type,
            'start_time': self.start_time,
        }

    @property
    def mouse_id(self):
        return self._trial_info['mouse_id']

    @property
    def mouse_name(self):
        return database.fetch_mouse(self.mouse_id)['mouse_name']

    @property
    def start_time(self):
        return self._trial_info['start_time']

    @property
    def stop_time(self):
        return self._trial_info['stop_time']

    @property
    def filename(self):
        return self._trial_info['behavior_file']

    def __setattr__(self, item, value):
        super().__setattr__(item, value)
        if item[0:4] == '_db_':
            # Setting an attribute prefixed with `_db_` will flag it for a
            # database update if self.save is called
            # This is a work-around to avoid namespace polution. Generally
            # variables in self._trial_info should be exposed via properties
            # as needed, and protected with setters to sanitize changes
            self._trial_info[item[4:]] = value

    @property
    def archived(self):
        # Fix this! It will break as-is. What is it even for?

        try:
            return self._props['archived']
        except KeyError:
            if self.get('tSeriesDirectory') is None:
                self._props['archived'] = False
            else:
                self._props['archived'] = \
                    len(glob.glob(os.path.join(
                        self.tSeriesDirectory, '*.archive'))) > 0

            return self.archived

    def save(self, store=False):
        """Update the sql database trial information based on the keys of
        `self._trial_info`. Note you shouldn't normally read or write to this
        dictionary directly, but rather expose variables as needed by defining
        new properties. See below for example implementations.

        Parameters
        ----------
        store : bool, optional
            Whether to actually update the database. Defaults to False, which
            will just print the changes that would be made.

        Examples
        --------
        Expose a key in `self._trial_info` as a property with a corresponding
        setter method:

        >>> class DumbyExpt(BehaviorExperiment):
        >>>     @property
        >>>     def my_awesome_attribute(self):
        >>>         return self._trial_info['my_awesome_attribute']
        >>>     @my_awesome_attribute.setter
        >>>     def my_awesome_attribute(self, value):
        >>>         self._trial_info['my_awesome_attribute'] = value

        Alternatively, attributes prefixed with `_db_` will automatically
        edit the corresponding value in `self._trial_info` when they are set:

        >>>     @my_awesome_attribute.setter
        >>>     def my_awesome_attribute(self, value):
        >>>         # this updates self._trial_info['my_awesome_attribute'] !
        >>>         self._db_my_awesome_attribute = value

        Set the attribute value and update the database:

        >>> expt = DumbyExpt(1)
        >>> expt.my_awesome_attribute = 'foo'
        >>> expt.save(store=True)
        """

        # compare to the clean dict created during init to find edits
        updates = {k: v for k, v in self._trial_info.items() if k not in
                   self._trial_info_clean.keys()
                   or v != self._trial_info_clean[k]}

        update_trial = False
        trial_args = ['behavior_file', 'mouse_name', 'start_time', 'stop_time',
                      'experiment_group']

        if not store:
            print(f'changes to {self.trial_id}: {updates}')
        else:
            print(f'saving changes to {self.trial_id}: {updates}')
            for key, value in updates.items():
                if key == 'trial_id':
                    raise Exception('trial_id cannot be changed')
                elif key in trial_args and key != 'mouse_name':
                    update_trial = True
                else:
                    if value is None:
                        database.delete_trial_attr(self.trial_id, key)
                    else:
                        if isinstance(value, dict) or isinstance(value, list):
                            value = json.dumps(value)
                        database.update_trial_attr(self.trial_id, key, value)

            self._trial_info_clean = self._trial_info.copy()

        if update_trial:
            database.update_trial(*[self.get(k) for k in trial_args],
                                  trial_id=self.trial_id)

    def delete(self):
        database.delete_trial(self.trial_id)
        self._trial_info = {}

    @classmethod
    def create(cls, tdml_path, repickle=False, overwrite=False):
        """Parse and pickle a tdml behavior file and create a database entry,
        returning the corresponding BehaviorExperiment object. If the
        experiment already exists in the database, simply return the
        BehaviorExperiment object, unless overwrite is passed.

        Parameters
        ----------
        tdml_path : str
            Path to tdml behavior file from which to create the experiment
        repickle : bool, optional
            Whether to reparse and pickle tdml file, if pickle file already
            exists.
        overwrite : bool, optional
            Delete and recreate the database entry for this behavior file.
            Defaults to False

        Returns
        -------
        Instance of BehaviorExperiment
        """

        assert (tdml_path.endswith('.tdml') and os.path.exists(tdml_path)), \
            "Not a valid tdml file"
        pkl_path = tdml_path.replace('.tdml', '.pkl')

        # make sure tdml file is pickled, re-pickling if desired
        if os.path.exists(pkl_path) and not repickle:
            pass
        else:
            from lab3.misc.tdml_pickler import pickle_file
            pickle_file(tdml_path)

        # (re)create database entry, or just get the trial_id if not overwrite
        trial_id = database.add_experiment_to_database(
            pkl_path, overwrite=overwrite)

        return cls(trial_id)

    def repickle_tdml(self):
        """Re-parse and pickle the tdml file associated with this experiment"""
        from lab3.misc.tdml_pickler import pickle_file
        pickle_file(self.filename.replace('.pkl', '.tdml'))

    def pair_imaging_data(self, sima_path, force_pairing=False, store=True):
        """Pair experiment data with imaging data from a sima folder,
        updating the database and returning the corresponding
        ImagingExperiment object.

        Parameters
        ----------
        sima_path : str
            Path to sima folder for imaging data
        force_pairing : bool, optional
            If this experiment is already paired with a sima folder, whether to
            overwrite pairing. Default is False
        store : bool, optional
            Whether to store the newly paired sima_path in the database.
            Defaults to True.
            """
        return ImagingExperiment(self.trial_id, sima_path,
                                 force_pairing=force_pairing,
                                 store=store)

    @property
    def behavior_data(self):
        """Get unformatted behavior dictionary from pkl file."""
        with open(self.filename.replace('.tdml', '.pkl'), 'rb') as f:
            data = pkl.load(f, encoding='latin1')
        return data

    @functools.lru_cache(maxsize=5)
    def format_behavior_data(self, sampling_interval=0.1, discard_pre=0,
                             discard_post=np.inf, sigma=0.1):
        """Format behavior dictionary. Interval variables are converted to
        boolean vectors that denote activity at each sampling period.
        Continuous variables are re-sampled at each discrete sampling period.

        Parameters
        ----------
        sampling_interval : float, optional
            Sampling rate for discretizing time. Variables expressed as
            intervals (i.e. vectors of start and stop times) are converted to
            binary vectors that are True for frames inside the intervals.
            Continuous variables are resampled at the corresponding time
            points.
        discard_pre : {'first_lap', float}, optional
            If 'first_lap', truncate data before the first complete lap. If
            float, truncate data occuring before this time. Default behavior
            returns all data points
        discard_post : float, optional
            Truncate data occuring after this time. Default behavior returns
            all data points
        sigma : float, optional
            Standard deviation of gaussian smoothing kernel for smoothing
            velocity variable, in seconds. Defaults to 0.1

        Returns
        -------
        beh_dict : dict
            Behavior data dictionary with variables expressed in discrete time
            bins

        See also
        --------
        experiment.utils.discretize_behavior
        """

        return utils.discretize_behavior(self.behavior_data,
                                         sampling_interval=sampling_interval,
                                         discard_pre=discard_pre,
                                         discard_post=discard_post,
                                         sigma=sigma)

    def velocity(self, sigma=0.1, **beh_kws):
        """Get velocity trace.

        Parameters
        ----------
        sigma : float, optional
            Standard deviation of the gaussian smoothing kernel to apply to the
            velocity trace, in seconds. Defaults to 0.1.

        Keyword arguments are passed directly to format_behavior_data, and can
        be used to modify sampling rate and trimming.
        """

        return self.format_behavior_data(sigma=sigma, **beh_kws)['velocity']

    def discrete_position(self, num_bins=100, **beh_kws):
        """Get discrete position bin for each behavior sample

        Parameters
        ----------
        num_bins : int, optional
            Divide track into this number of discrete position bins. Defaults
            to 100

        Keyword arguments are passed directly to format_behavior_data, and can
        be used to modify sampling rate and trimming.
        """

        continuous_position = self.format_behavior_data(
            **beh_kws)['treadmillPosition']
        return np.floor(continuous_position * num_bins).astype('int')

    def laps(self, **beh_kws):
        """Get lap number for each behavior sample

        Keyword arguments are passed directly to format_behavior_data, and can
        be used to modify sampling rate and trimming.
        """

        return self.format_behavior_data(**beh_kws)['lap_bin']

    def behavior_signals(self, label, signal_type=None, channel=None, 
                         as_time_signals=False, sampling_interval=0.1, 
                         **kwargs):
        behavior_signals = self.format_behavior_data(
            sampling_interval=sampling_interval, **kwargs)
        signals = behavior_signals[label]
        signals = pd.DataFrame(signals, columns=[label]).T

        if as_time_signals:
            signals.columns = \
                signals.columns * behavior_signals['sampling_interval']
                
        return signals

    def behavior_events(self, event_type, **kwargs):
        return BehaviorEventGroup.load(self, event_type, **kwargs)

    def signals(self, *args, source='behavior', **kwargs):
        return super().signals(*args, source=source, **kwargs)        

class ImagingMixin:

    """Mixin for 2-photon imaging experiments. Provides methods for accessing
    and manipulating extracted imaging signals."""

    @property
    def frame_rate(self):
        return 1 / self.frame_period

    @property
    def frame_period(self):
        return self.imaging_parameters['frame_period']

    @property
    def imaging_system(self):
        return self.imaging_parameters['imaging_system']

    @property
    def imaging_parameters(self):
        """Returns a dictionary of imaging parameters from the h5 file"""
        try:
            return self._params
        except (KeyError, AttributeError):
            params_file = os.path.join(self.sima_path, "imaging_parameters.json") 
            try:
                with open(params_file) as f:
                    self._params = json.load(f)
                return self._params
            except FileNotFoundError:
                from .utils import NumpyEncoder
                h5 = h5py.File(self.imaging_dataset.sequences[0]._path, 'r')
                self._params = {k: v for k, v in h5['/imaging'].attrs.items()}

                with open(params_file, "w") as f:
                    json.dump(self._params, f, cls=NumpyEncoder)

                return self._params

    @property
    def imaging_dataset(self):
        """Return the sima ImagingDataset associated with this experiment"""
        return ImagingDataset.load(self.sima_path)

    def imaging_times(self, max_frame='trim_to_behavior'):

        if max_frame == 'trim_to_behavior':
            # figure out length of behavior data in imaging frames
            beh_dict = self.format_behavior_data()
            max_frame = beh_dict['recordingDuration'] / self.frame_period
        elif max_frame is None:
            max_frame = self.imaging_dataset.num_frames  #self.signals().shape[1] (shouldn't this be the same?!)
        return np.arange(max_frame)*self.frame_period

    @property
    def sima_path(self):
        """Return the sima directory paired to this experiment"""
        return self._db_sima_path

    @sima_path.setter
    def sima_path(self, path):
        path = os.path.normpath(path)
        assert path.endswith('.sima') and os.path.isdir(path), \
            "Not a valid sima path"
        self._db_sima_path = path

    @property
    def signals_path(self):
        return os.path.join(self.sima_path, "signals.h5")

    @property
    def imaging_events_path(self):
        return os.path.join(self.sima_path, "events.h5")

    @property
    def suite2p_imaging_dataset(self, wrapper_class=None):
        """Cast SIMA dataset as a Suite2p strategy.

        Parameters
        ----------
        wrapper_class : Suite2pStrategy, optional
            Class with which to instantiate the Suite2p imaging dataset.
            Defaults to the standard
            :class:`~lab3.extraction.s2p.Suite2pImagingDataset` class.
            Override to use your own custom subclass.
        """

        if wrapper_class is None:
            from ..extraction.s2p import Suite2pImagingDataset
            wrapper_class = Suite2pImagingDataset

        return wrapper_class.load(self.sima_path)

    def signals_file(self, mode='r', **kwargs):
        """Returns the HDFStore containing extracted and processed signals,
        read only"""
        return SignalFile(self.signals_path, mode=mode,  **kwargs)

    @default_label_handling
    def imaging_signals(self, signal_type='raw', channel='Ch2', 
                        as_time_signals=False, label=None, max_frame=None, 
                        z_score=False):
        """Retrieve signals from signals.h5.

        Parameters
        ----------
        signal_type : str {'raw', 'npil', 'dfof', 'spikes', or custom}
            Name of signal type, defaults to 'raw'
        channel : {str, int}, optional
            Channel name or index in the imaging dataset
        label : str, optional
            ROI list label in the imaging dataset. If None, the most recently
            accessed label in the imaging dataset will be used
        max_frame : int, optional
            Only return the first max_frame frames. Defaults to None (all
            frames are returned)
        z_score : bool or str {'nonparametric'}, optional
            Return the z-scored signal. If 'nonparametric', will use median and
            IQR in lieu of mean and stdev
        Returns
        -------
        signals : pd.DataFrame
        """

        with self.signals_file() as signal_file:
            signals = signal_file.get(f"/{channel}/{label}/{signal_type}")

        if z_score is True:
            signals = signals.apply(nan_zscore, axis=1)
        elif z_score == 'nonparametric':
            signals = signals.apply(nonparametric_nan_zscore, axis=1)

        if as_time_signals:
            signals.columns = self.imaging_times(max_frame=max_frame)

        return signals

    def signals(self, *args, source='imaging', **kwargs):
        return super().signals(*args, source=source, **kwargs)

    def raw_fluorescence(self, **kwargs):
        return self.signals(signal_type='raw', **kwargs)

    def dfof(self, **kwargs):
        return self.signals(signal_type='dfof', **kwargs)

    def spikes(self, **kwargs):
        return self.signals(signal_type='spikes', **kwargs)

    def check_overwrite(self, store_key, overwrite):
        with self.signals_file() as signals_file:
            key_exists = store_key in signals_file
            assert not key_exists or overwrite, \
                f"Key {store_key} already exists in signal file. Set " \
                + "overwrite=True to recalculate"

    def imaging_events(self, event_type, channel='Ch2', label=None, **kwargs):
        """Retrieve imaging events from events.h5"""
        if 'transient' in event_type:
            return ExperimentTransientGroup.load(self, event_type=event_type, 
                                            channel=channel, label=label, **kwargs)
        else:
            # This is probably incorrect for imaging-events since it does differentiate by cell
            return EventGroup.load(self, event_type=event_type,
                               channel=channel, label=label, **kwargs)

    @default_label_handling
    def calculate_signals(self, strategy, channel='Ch2', label=None,
                          to_type='misc', overwrite=False, **kwargs):
        """Run a signal processing analysis.

        Parameters
        ----------
        strategy : lab3.core.Automorphism
            Instance of a valid signal analysis.
        channel : {str, int}, optional
            Channel label or index in imaging dataset
        label : str, optional
            ROI label in imaging dataset. If None, the most recently accessed
            label in the imaging dataset will be used
        to_type : str, optional
            Name of signal type, for storing in signals.h5. The store key
            for the results will become channel/label/to_type.
        overwrite : bool, optional
            Whether to overwrite signal key if it exists already. Defaults to
            False
        **kwargs
            Additional keyword arguments are passed directly to
            strategy.apply_to()

        Returns
        -------
        pd.DataFrame
        """

        store_key = f"/{channel}/{label}/{to_type}"
        self.check_overwrite(store_key, overwrite)

        # replace this with a check for a more specific SignalAnalysis class
        from lab3.core import Automorphism
        assert isinstance(strategy, Automorphism), \
            "strategy must be an instance of an Automorphism class!"

        s = self.apply(strategy, channel=channel, label=label, **kwargs)

        with self.signals_file(mode='a') as signals_file:
            signals_file.put(store_key, s)

        return s

    def infer_spikes(self, strategy, **kwargs):
        """Run spike inferencing on signals.

        Parameters
        ----------
        strategy : SpikesStrategy
            Instance of a SpikesStrategy. See lab3.signal.spikes
        channel : {str, int}, optional
            Channel label or index in imaging dataset
        label : str, optional
            ROI label in imaging dataset. If None, the most recently accessed
            label in the imaging dataset will be used
        from_type : str {'dfof', 'raw'} or custom, optional
            Signal type to perform inference on. Defaults to 'dfof'
        overwrite : bool, optional
            Whether to overwrite signal key if it exists already. Defaults to
            False

        Returns
        -------
        spikes : pd.DataFrame
        """

        # TODO should we check to make sure this is a SpikesStrategy?
        assert strategy.name == 'spikes'
        return self.calculate_signals(strategy, to_type='spikes', **kwargs)

    def calculate_dfof(self, strategy, **kwargs):
        """Calculate dF/F from raw signals.

        Parameters
        ----------
        strategy : DFOFStrategy
            Instance of a DFOFStrategy. See lab3.signal.dfof
        channel : {str, int}, optional
            Channel label or index in imaging dataset
        label : str, optional
            ROI label in imaging dataset. If None, the most recently accessed
            label in the imaging dataset will be used
        overwrite : bool, optional
            Whether to overwrite signal key if it exists already. Defaults to
            False

        Returns
        -------
        dfof : pd.DataFrame
        """

        # TODO should we check to make sure this is a DFOFStrategy?
        assert strategy.name == 'DFOF'
        return self.calculate_signals(strategy, to_type='dfof', **kwargs)

    @default_label_handling
    def roi_list(self, label=None):
        """Get a list of ROI objects

        Parameters
        ----------
        label : str, optional
            Label of the desired ROI list. If None, the most recently accessed
            label in the imaging dataset will be used

        Returns
        --------
        sima.ROI.ROIList
            ROIList object containing all the ROIs for the passed label
        """

        # TODO default handling? or does SIMA handle this?
        return self.imaging_dataset.ROIs[label]

    def delete_roi_list(self, channel='Ch2', label=None, do_nothing=True):
        """Remove an existing ROI list. This both removes the ROIs from the
        underlying sima ImagingDataset (i.e. deletes them from
        folder.sima/rois.pkl), and also deletes any corresponding entries in
        signals.h5.

        Parameters
        ----------
        channel : {str, int}, optional
        label : str, optional
        do_nothing : bool, optional
        """

        if label in self.imaging_dataset.ROIs.keys():
            signals_file = self.signals_file(mode='a')
            keys_to_delete = [k for k in signals_file.keys()
                              if f'/{label}/' in k]
            print(f"Deleting keys:\n{keys_to_delete} "
                  + f"from {signals_file._path}")

            if not do_nothing:
                for key in keys_to_delete:
                    signals_file.remove(key)
                self.imaging_dataset.delete_ROIs(label=label)

            signals_file.close()

        else:
            raise KeyError(f"Label '{label}' is not in the imaging dataset")

    @default_label_handling
    def calculate_transients(self, channel='Ch2', label=None, overwrite=False,
                             strategy=None):
        raise NotImplementedError


class ImagingExperiment(ImagingMixin, BehaviorExperiment):

    """Imaging experiment with behavior. If the imaging directory is not
    already paired with the database trial id, or you would like to modify it,
    this can be passed during initialization and the database will be updated
    accordingly.

    Parameters
    ----------
    trial_id : int
        ID of experiment in the sql database
    sima_path : str, optional
        Path to a sima folder. If passed, this imaging data will be paired in
        the database with this trial_id.
    force_pairing : bool, optional
        If trial_id is already paired with a sima folder but sima_path
        is passed, whether to overwrite pairing. Default is False
    store : bool, optional
        Whether to store the newly paired sima_path in the database, if passed.
        Defaults to False.
    """

    def __init__(self, trial_id, sima_path=None, force_pairing=False,
                 store=False):

        super().__init__(trial_id)

        if sima_path is not None:
            try:
                if self._trial_info['sima_path'] and not force_pairing:
                    raise ValueError(
                        f"Trial {trial_id} is already paired to imaging"
                        + f" data {self._trial_info['sima_path']}. "
                        + "Set force_pairing=True to force pairing")
            except (KeyError, AttributeError):
                pass
            self.sima_path = sima_path
            self.save(store=store)
        else:
            try:
                self.sima_path = self._trial_info['sima_path']
                self.imaging_dataset
            except Exception:
                raise AttributeError(
                    f"Trial {trial_id} has no sima_path, or the path "
                    + "is not valid. Try passing a sima_path when "
                    + f"initializing {type(self).__name__}.")

    def format_behavior_data(self, image_sync=True, **kwargs):
        """Format behavior dictionary. Interval variables are converted to
        boolean vectors that denote activity at each sampling period.
        Continuous variables are re-sampled at each discrete sampling period.
        By default, behavior is binned to synchronize with the imaging data.

        Parameters
        ----------
        image_sync : bool, optional
            If True, sampling_interval is set to imaging frame period, and
            the length of behavior and imaging data are matched. Defaults to
            True.
        sampling_interval : float, optional
            Sampling rate for discretizing time. Variables expressed as
            intervals (i.e. vectors of start and stop times) are converted to
            binary vectors that are True for frames inside the intervals.
            Continuous variables are resampled at the corresponding time
            points. Ignored if image_sync is True.
        discard_pre : {'first_lap', float}, optional
            If 'first_lap', truncate data before the first complete lap. If
            float, truncate data occuring before this time. Default behavior
            returns all data points. Ignored if image_sync is True.
        discard_post : float, optional
            Truncate data occuring after this time. Default behavior returns
            all data points. Ignored if image_sync is True.
        sigma : float, optional
            Standard deviation of gaussian smoothing kernel for smoothing
            velocity variable, in seconds. Defaults to 0.1

        Returns
        -------
        beh_dict : dict
            Behavior data dictionary with variables expressed in discrete time
            bins

        See also
        --------
        experiment.utils.discretize_behavior
        """

        if image_sync:
            sigma = kwargs.pop('sigma', 0.1)
            if len(kwargs):
                warnings.warn("image_sync is True (default), so additional "
                              + "keyword arguments were ignored.")

            kwargs.update({'sampling_interval': self.frame_period,
                           'discard_post': self.frame_period
                           * self.imaging_dataset.num_frames,
                           'discard_pre': 0,
                           'sigma': sigma})

        return {**super().format_behavior_data(**kwargs),
                **{'image_sync': image_sync}}

    def velocity(self, image_sync=True, sigma=0.1, **beh_kws):
        """Get velocity trace.

        Parameters
        ----------
        image_sync : bool, optional
            If True, sampling_interval is set to imaging frame period, and
            the length of behavior and imaging data are matched. Defaults to
            True.
        sigma : float, optional
            Standard deviation of the gaussian smoothing kernel to apply to the
            velocity trace, in seconds. Defaults to 0.1.

        Additional keyword arguments are passed directly to
        format_behavior_data, and are used to override defaults only if
        image_sync is False.

        Returns
        -------
        velocity : array (n_samples,)
        """

        return super().velocity(image_sync=image_sync, sigma=sigma, **beh_kws)

    def signals(self, max_frame='trim_to_behavior', **kwargs):
        """Retrieve signals from signals.h5. By default, only return frames
        recorded with behavior data.

        Parameters
        ----------
        max_frame : int or 'trim_to_behavior', optional
            Only include the first max_frame frames. If 'trim_to_behavior',
            trim the excess frames to match behavior data length. Defaults to
            'trim_to_behavior'.
        channel : {str, int}, optional
            Channel name or index in the imaging dataset
        label : str, optional
            ROI list label in the imaging dataset
        signal_type : str {'raw', 'npil', 'dfof', 'spikes', or custom}
            Name of signal type, defaults to 'raw'

        Returns
        -------
        signals : pd.DataFrame
        """

        if max_frame == 'trim_to_behavior':
            # figure out length of behavior data in imaging frames
            beh_dict = self.format_behavior_data()
            max_frame = int(beh_dict['recordingDuration'] / self.frame_period)

        return super().signals(max_frame=max_frame, **kwargs)


class ImagingOnlyExperiment(ImagingMixin, _BaseExperiment):

    """Imaging experiment with no behavior / database records. To initialize,
    pass a SIMA directory rather than a trial ID.

    Parameters
    ----------
    sima_path : str
    """

    def __init__(self, sima_path):

        super().__init__(trial_id=sima_path)
        self.sima_path = sima_path


class LFPMixin:

    """Mixin for LFP experiments"""

    @property
    def lfp_path(self):
        """Return the location of the LFP data"""
        return self._lfp_path

    @lfp_path.setter
    def lfp_path(self, path):
        self._lfp_path = path

    @property
    def lfp_signals_path(self):
        return os.path.join(self.lfp_path, "lfp_signals.h5")

    def lfp_signals_file(self, mode='r', **kwargs):
        return SignalFile(self.lfp_signals_path, mode=mode, **kwargs)

    def lfp_signals(self, label, channel=None):
        """Retrieve signals from lfp_signals.h5.

        Parameters
        ----------
        label : str, optional
            Something like 'LFP', 'EMG', etc.
        channel : {str, int}, optional
            Channel name or index in the LFP dataset
        signal_type : str {'raw', 'filtered', or custom}
            Name of signal type, defaults to 'raw'

        Returns
        -------
        signals : pd.DataFrame
        """

        with self.lfp_signals_file() as signal_file:
            signals = signal_file.get(f"{label}")
        if channel is not None:
            signals = signals.iloc[channel]
        return signals

    @property
    def lfp_events_path(self):
        return os.path.join(self.lfp_path, 'lfp_events.h5')

    def lfp_events(self, event_type, label='LFP', channel=None, **kwargs):
        return LFPEventGroup.load(self, event_type=event_type,
                                  label=label, channel=channel, **kwargs)

    def check_overwrite(self, store_key, overwrite):
        with self.lfp_signals_file() as signals_file:
            key_exists = store_key in signals_file
            assert not key_exists or overwrite, \
                f"Key {store_key} already exists in signal file. Set " \
                + "overwrite=True to recalculate"


class LFPExperiment(LFPMixin, BehaviorExperiment):

    """LFP experiment with behavior but no imaging data"""

    def __init__(self, trial_id):
        super().__init__(trial_id)


class LFPOnlyExperiment(LFPMixin, _BaseExperiment):

    """LFP expreriment with no behavior or imaging data"""

    def __init__(self, lfp_path):
        super().__init__(trial_id=lfp_path)
        self.lfp_path = lfp_path


class ImagingLFPExperiment(LFPMixin, ImagingExperiment):

    """Combination 2p imaging and LFP experiment with behavior"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.lfp_path is None:
            self.lfp_path = os.path.join(os.path.split(
                                self.sima_path)[0], 'LFP')

    @property
    def imaging_times(self, **kwargs):
        """Attempt to timelock to LFP

        """
        nframes = self.signals().shape[1]

        try:
            frames = self.events(signal_type='LFP', event_type='imaging_frame')
            return np.array(frames.onset)[:nframes]
        except Exception:  # what error are we catching here?!
            print("Failed to timelock to LFP!")
            return super().imaging_times(**kwargs)


class ImagingLFPOnlyExperiment(LFPMixin, ImagingOnlyExperiment):
    pass


# for populating the documentation website
__all__ = [
    'BehaviorExperiment',
    'ImagingMixin',
    'ImagingExperiment',
    'ImagingOnlyExperiment',
    'LFPMixin',
    'LFPExperiment',
    'LFPOnlyExperiment',
    'ImagingLFPExperiment',
    'ImagingLFPOnlyExperiment',
    'fetch_trials',
]

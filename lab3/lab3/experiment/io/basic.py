"""Simple tools for data sharing"""

# TODO: it would be nice to have experiment classes that can read from these
# files so that the same analysis can be done without the database
# Right now we just handle simple exports

import os

import pandas as pd


class H5DataDumper:

    """This object takes experiments and dumps them to HDFStores.

    For now this only works on ImagingExperiments, but we'll make it more
    flexible in the future with options about which data/metadata to store.

    Parameters
    ----------
    savepath : str
        Where to save the h5 stores
    binarize : bool, optional
        If dumping spikes, whether tto binarize signals first.
    """

    def __init__(self, savepath, binarize=True):

        self.savepath = savepath
        self.binarize = binarize

    @property
    def savepath(self):
        return self._savepath

    @savepath.setter
    def savepath(self, p):
        if not os.path.exists(p):
            os.mkdir(p)
        self._savepath = p

    def _dump_imaging_data(self, expt, store):

        for key in expt.signals_file().keys():
            # what happens if there are no signals? need error handling here
            # for non-imaging experiments
            _, channel, label, signal_type = key.split(os.sep)

            # we retrieve the signals via the method so that, if we're dealing
            # with an imaging experiment, it gets trimmed to match the behavior
            # under the hood
            signal = expt.signals(channel=channel, label=label,
                                  signal_type=signal_type)
            if (signal_type == 'spikes') & self.binarize:
                signal = signal.gt(0)

            store.put(key, signal)

    def _dump_behavior_data(self, expt, store):

        behavior = {}
        for key, value in expt.format_behavior_data().items():
            try:
                # TODO - handle max frame when not ImagingMixin
                if len(value) == len(expt.imaging_times()):
                    behavior[key] = value
            except Exception:  # TypeError?
                pass

        behavior = pd.DataFrame(behavior)
        store.put('behavior', behavior)

    def _dump_metadata(self, expt, store):
        # TODO
        pass

    def _datadump(self, expt, store):
        self._dump_imaging_data(expt, store)
        self._dump_behavior_data(expt, store)
        self._dump_metadata(expt, store)

    def dump(self, expt, overwrite=False):
        """Given an experiment object, create the h5 store"""
        h5_path = os.path.join(self.savepath, f'{expt.trial_id}.h5')
        if os.path.isfile(h5_path):
            if overwrite:
                os.remove(h5_path)
            else:
                print(f"'{h5_path}' exists and overwrite=False, skipping...")
                return

        with pd.HDFStore(h5_path) as store:
            self._datadump(expt, store)


class VRDataDumper(H5DataDumper):

    def _dump_behavior_data(self, expt, store):

        behavior = {
            'velocity': expt.velocity(),
            'position': expt.discrete_position(),
            'iti': expt.iti(),
            'valid': expt.valid_samples(),
            'reward': expt.format_behavior_data()['reward'],
            'licking': expt.format_behavior_data()['licking'],
            'lap': expt.format_behavior_data()['lap_bin'],
        }
        behavior = pd.DataFrame(behavior)
        store.put('behavior', behavior)

    def _dump_masks(self, expt, store):
        masks = pd.DataFrame(
            [{'roi_label': roi.label, 'roi_mask': roi.mask[0]}
             for roi in expt.roi_list()]).set_index('roi_label')
        store.put('roi_masks', masks)

    def _dump_timeavg(self, expt, store):
        ds = expt.suite2p_imaging_dataset
        time_avg = ds.ops_list[0]['meanImg']
        store.put('time_avg', pd.DataFrame(time_avg))

    def _datadump(self, expt, store):
        super()._datadump(expt, store)
        self._dump_masks(expt, store)
        self._dump_timeavg(expt, store)

    def dump(self, expt, overwrite=False):
        from lab3.experiment.virtual import VirtualImagingExperiment
        assert isinstance(expt, VirtualImagingExperiment)
        super().dump(expt, overwrite=overwrite)

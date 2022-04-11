"""Convenience classes for converting old data to the new signal storage
format in LAB3.

Note these methods don't actually work... what is `unpickle_old`?!"""

import os
import pandas as pd
import pickle as pkl

from lab3.experiment.base import ImagingMixin, ImagingOnlyExperiment


class OldImagingMixin(ImagingMixin):

    def signals(self, signal_type='raw', channel='Ch2', label=None,
                max_frame=None):
        channel_idx = self.imaging_dataset._resolve_channel(channel)
        if signal_type == 'raw':
            signals = self._load_old_signals(channel=channel_idx,
                                             label=label)
        elif signal_type == 'dfof':
            signals = self._load_old_dfof(channel=channel_idx,
                                          label=label)
        else:
            raise NotImplementedError(
                f'Unsupported signal type `{signal_type}`')

        if max_frame is not None:
            signals = signals.iloc[:, :(max_frame - 1)]

        return signals

    def write_newstyle(self, signal_type='raw', channel='Ch2',
                       label=None, **kwargs):
        signals = self.signals(signal_type=signal_type, channel=channel,
                               label=label, **kwargs)
        signal_key = f"/{channel}/{label}/{signal_type}"
        with self.signals_file('a') as signal_file:
            signal_file.put(signal_key, signals)

    def write_all(self, channel='Ch2'):
        channel_idx = self.imaging_dataset._resolve_channel(channel)
        signals_file = os.path.join(self.sima_path, f'signals_{channel_idx}.pkl')
        dfof_file = os.path.join(self.sima_path, f'dFoF_{channel_idx}.pkl')

        raw_data = unpickle_old(signals_file)
        dfof_data = unpickle_old(dfof_file)

        for label, data in raw_data.items():
            try:
                roi_ids = pd.Index([r['label'] for r in data['rois']], name='roi_label')
                raw = data['raw'][0]
                signal_key = f"/{channel}/{label}/raw"
                signals = pd.DataFrame(raw, index=roi_ids)
                with self.signals_file('a') as signal_file:
                    print(f"Writing {signal_key}...")
                    signal_file.put(signal_key, signals)

                # Is there dfof? If so, write
                dfof = dfof_data[label]['traces'][...,0]
                signal_key = f"/{channel}/{label}/dfof"
                with self.signals_file('a') as signal_file:
                    print(f"Writing {signal_key}...")
                    signal_file.put(signal_key, pd.DataFrame(dfof, index=roi_ids))
            except Exception as exc:
                print(f"Failed to write `{signal_key}` because of {exc}")

    def _load_old_signals(self, channel=0, label=None):
        signals_file = os.path.join(self.sima_path, f'signals_{channel}.pkl')
        data = unpickle_old(signals_file)[label]
        roi_ids = pd.Index([r['label'] for r in data['rois']], name='roi_label')
        raw = data['raw'][0]
        return pd.DataFrame(raw, index=roi_ids)

    def _load_old_dfof(self, channel=0, label=None):
        signals_file = os.path.join(self.sima_path, f'signals_{channel}.pkl')
        data = unpickle_old(signals_file)[label]
        roi_ids = pd.Index([r['label'] for r in data['rois']], name='roi_label')

        dfof_file = os.path.join(self.sima_path, f'dFoF_{channel}.pkl')
        data = unpickle_old(dfof_file)[label]
        dfof = data['traces'][..., 0]
        return pd.DataFrame(dfof, index=roi_ids)


class OldImagingExperiment(OldImagingMixin, ImagingOnlyExperiment):
    """Useful for expeditious conversion to newstyle.
    Probably *not* so good for doing extensive analysis on its own

    TODO : stick an example here on how to use objects to convert old data
    """
    pass

def unpickle_old(filename):
    with open(filename, 'rb') as f:
        data = pkl.load(f, encoding='latin1')
    return data


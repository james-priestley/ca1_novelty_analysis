import os

import numpy as np
import pandas as pd
import rhd

from lab3.signal import SignalFile


# label, channel, signal_type
def overwrite_lfp(base_key, overwrite, signal_file):
    if base_key in signal_file:
        assert overwrite, f"Signals already exist for {base_key}. " \
            + "Set overwrite=True to re-import.\nCurrent file structure:" \
            + f" \n {signal_file.info()}"

        print(f"Deleting previous import at {base_key}")
        signal_file.remove(base_key)


def trim_to_imaging(rhd_data):
    try:
        rhd_data['trimmed']
        return rhd_data
    except KeyError:
        expt_start = np.where(rhd_data['board_adc_data'][1] > 1)[0][0]

        # Signals
        rhd_data['amplifier_data'] = rhd_data['amplifier_data'][:, expt_start:]
        rhd_data['board_adc_data'] = rhd_data['board_adc_data'][:, expt_start:]

        # Times
        rhd_data['t_amplifier'] = rhd_data['t_amplifier'][expt_start:]
        rhd_data['t_board_adc'] = rhd_data['t_board_adc'][expt_start:]

        # Re-zero
        rhd_data['t_amplifier'] -= rhd_data['t_amplifier'][0]
        rhd_data['t_board_adc'] -= rhd_data['t_board_adc'][0]

        rhd_data['trimmed'] = True

        return rhd_data


def import_intan_to_lfp_signals_file(
        intan_path, lfp_signals_path, label='LFP', preprocess_func=None,
        amplifier_channels=None, adc_channels=None, overwrite=False):
    """Import rhd to an lfp_signals.h5

    Parameters
    ----------
    intan_path : str
        Path to .rhd file
    lfp_signals_path : str
        Directory where LFP data is stored
    preprocess_func : callable, optional
        Preprocessing on output of rhd.read_data. This should take a
        dictionary as input and return the same dictionary as output, with
        the preprocessing you want performed.
    label : str, optional
        Label to be applied to the 'amplifier_data' channels (ignored if
        'amplifier_channels' is specified)
    amplifier_channels : iterable, optional
        Labels to interpret amplifier channels as
    adc_channels : iterable, optional
        Labels to interpret ADC channels as
    overwrite : bool
        Whether to overwrite existing data
    """

    data = rhd.read_data(intan_path)

    if preprocess_func is not None:
        data = preprocess_func(data)

    with SignalFile(os.path.join(lfp_signals_path,
                                 'lfp_signals.h5')) as signal_file:
        print("Storing...")
        if amplifier_channels is None:
            overwrite_lfp(label, overwrite, signal_file)

            print(f"Deleting previous import at {base_key}")
            signal_file.remove(base_key)

            raw = pd.DataFrame(data['amplifier_data'],
                               columns=data['t_amplifier'],
                               index=np.arange(len(data['amplifier_data'])))

            signal_file.put(f"{label}/raw", raw)
        else:
            raw = pd.DataFrame(data['amplifier_data'],
                               columns=data['t_amplifier'],
                               index=amplifier_channels)
            for channel_name in raw.index:
                overwrite_lfp(channel_name, overwrite, signal_file)
                signal_file.put(f"{channel_name}/raw",
                                raw.loc[channel_name].to_frame(channel_name))

        if adc_channels is None:
            overwrite_lfp("ADC", overwrite, signal_file)
            adc = pd.DataFrame(data['board_adc_data'],
                               columns=data['t_board_adc'],
                               index=np.arange(len(data['board_adc_data'])))
            signal_file.put(f"ADC/raw", adc)
        else:
            adc = pd.DataFrame(data['board_adc_data'],
                               columns=data['t_board_adc'],
                               index=adc_channels)
            for channel_name in adc.index:
                overwrite_lfp(channel_name, overwrite, signal_file)
                signal_file.put(f"{channel_name}/raw",
                                adc.loc[channel_name].to_frame(channel_name))
        # signal_file["metadata"] = data['frequency_parameters']
        print("Done!")

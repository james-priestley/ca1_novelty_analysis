"""Convenience functions for experiment classes"""

import os
from copy import deepcopy
import json

import numpy as np
import h5py
from scipy.interpolate import interp1d
from scipy.signal import gaussian, convolve
from sima.ROI import ROIList

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, bytes):
            return obj.decode()

        elif isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):

            return int(obj)

        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        
        elif isinstance(obj, (np.complex_, np.complex64, np.complex128)):
            return {'real': obj.real, 'imag': obj.imag}
        
        elif isinstance(obj, (np.bool_)):
            return bool(obj)

        elif isinstance(obj, (np.void)): 
            return None

        return json.JSONEncoder.default(self, obj)


def discretize_behavior(beh_dict, sampling_interval=0.1, discard_pre=0,
                        discard_post=np.inf, sigma=0.1):
    """Converts format of behavior data dictionary from continuous to discrete
    time. Intervals become boolean variables, and continuous variables are
    resampled at regular intervals.

    If there are additional variables to check for and retain in the
    behavior data, they should be added to this function.

    Parameters
    ----------
    sampling_interval : float, optional
        Sampling rate for discretizing time. Variables expressed as
        intervals (i.e. vectors of start and stop times) are converted to
        binary vectors that are True for frames inside the intervals.
        Continuous variables are resampled at the corresponding time points.
    discard_pre : {'first_lap', float}, optional
        If 'first_lap', truncate data before the first complete lap. If
        float, truncate data occuring before this time, in seconds. Default
        behavior returns all data points
    discard_post : float, optional
        Truncate data occuring after this time, in seconds. Default behavior
        returns all data points
    sigma : float, optional
        Standard deviation of gaussian smoothing kernel for smoothing
        velocity variable, in seconds. Defaults to 0.1

    Returns
    -------
    out : dict
        Behavior data dictionary with variables expressed in discrete time
        bins
    """

    # ---------------------------- PRELIMIANRIES ---------------------------- #

    total_duration = beh_dict['recordingDuration']
    out_dict = deepcopy(beh_dict)

    if discard_pre == 'first_lap':
        # the first lap never strictly starts at zero, so we have to build in
        # some tolerance
        try:
            FIRST_LAP_TOLERANCE = 0.001
            if beh_dict['treadmillPosition'][0, 1] > FIRST_LAP_TOLERANCE:
                discard_pre = beh_dict['lap'][0] / sampling_interval
            else:
                discard_pre = 0
        except KeyError:
            print("No lap information, continuing without!")

    # convert trim times to trim bins for slicing
    start = int(discard_pre / sampling_interval)
    stop = int(np.min([total_duration, discard_post]) / sampling_interval)
    out_dict['recordingDuration'] = (stop - start) * sampling_interval

    # ------------------------ DISCRETIZE VARIABLES ------------------------- #

    # convert interval variables
    out_dict.update(
        {k: interval_to_boolean(
                v, sampling_interval, total_duration
            )[start:stop]
         for k, v in out_dict.items() if is_interval(k, v)})

    # resample position
    out_dict.update({'treadmillPosition': continuous_to_discrete(
            out_dict['treadmillPosition'], sampling_interval, total_duration
        )[start:stop]})

    # ------------------------------ VELOCITY ------------------------------- #

    try:
        # newer behavior files should record dy explicitly
        dt, dy = out_dict['__position_updates'][:, 0:2].T

        # convert rotary encoder ticks to cm
        dy /= (out_dict['__trial_info']['position_scale'] * 10)

        # calculate velocity from dy/dt
        out_dict['velocity'] = calculate_velocity_from_updates(
            dy, dt, sampling_interval, total_duration, sigma=sigma)[start:stop]

    except KeyError:
        # otherwise calculate from the derivative of position
        out_dict['velocity'] = calculate_velocity(
            out_dict['treadmillPosition'], sampling_interval,
            out_dict['trackLength'], sigma=sigma)[start:stop]

    # ----------------------- MAKE NEW LAP VARIABLES ------------------------ #

    try:
        # find bin of each lap start (round up!)
        lap = [int(s - start) for s in
               np.ceil(beh_dict['lap'] / sampling_interval)
               if ((s >= start) and (s <= stop))]
        lap = np.unique([0] + lap)

        # create additional variable that gives the lap number in each
        # discrete bin
        lap_length = np.diff(np.append(lap, stop - start)).astype('int')
        lap_bin = np.concatenate(
            [[l] * n_bins for l, n_bins in enumerate(lap_length)])

        out_dict['lap'] = lap
        out_dict['lap_bin'] = lap_bin
    except KeyError:
        print("No lap information, continuing without!")

    # ------------------------------ CLEAN UP ------------------------------- #

    # log parameters
    out_dict.update({'sampling_interval': sampling_interval,
                     'discard_pre': discard_pre,
                     'discard_post': discard_post})

    # remove some extraneous information
    try:
        out_dict['json'] = out_dict['__trial_info']
    except KeyError:
        pass

    for k in list(out_dict.keys()):
        if '__' in k:
            out_dict.pop(k)

    return out_dict


def is_interval(key, val):
    """Helper function for finding interval variables from the standard behavior
    dictionary output"""

    if key in ['__position_y', 'treadmillPosition']:
        return False
    try:
        if val.shape[1] == 2:
            # intervals will have strictly two columns
            return True
        else:
            False
    except (AttributeError, IndexError):
        # either variable was not an array or did not have 2 dimensions
        return False


def interval_to_boolean(val, sampling_interval, total_duration):
    """Convert an interval variable to boolean vector

    Parameters
    ----------
    val : array of shape (n_samples, 2)
        Array of intervals. Each row is an event, where the first column gives
        the start time and the second column gives the stop time
    sampling_interval : float
        Width of each discrete time bin
    total_duration : int
        Sets the total length of the output vector, which will have length
        int(total_duration / sampling_interval)

    Returns
    -------
    out : array of length int(total_duration / sampling_interval)
        Boolean version of interval variable
    """
    out = np.zeros(int(total_duration / sampling_interval), 'bool')

    for i, (start, stop) in enumerate(val):

        # Is this still necessary if nans are fixed in tdmlPickler?
        if np.isnan(start):
            start = 0 if i == 0 else val[i-1, 0]
        if np.isnan(stop):
            try:
                stop = val[np.isfinite(val[:, 1])][i, 1]
            except IndexError:
                stop = len(out)

        start_frame = int(start / sampling_interval)
        stop_frame = int(stop / sampling_interval) + 1
        out[start_frame:stop_frame] = True

    return out


def continuous_to_discrete(val, sampling_interval, total_duration):
    """Resample a continuous variable (sampled at possibly variable intervals)
    to one sampled at regular, discrete time bins, via interpolation.

    Parameters
    ----------
    val : array of shape (n_samples, 2)
        Array of samples. Each row is a sample, where the first column gives
        the timestamp of the sample and the second column gives the value of
        the sample
    sampling_interval : float
        Width of each discrete time bin
    total_duration : int
        Sets the total length of the output vector, which will have length
        int(total_duration / sampling_interval)

    Returns
    -------
    out : array of length int(total_duration / sampling_interval)
        Resampled version of continuous variable
    """

    interp_func = interp1d(val[:, 0], val[:, 1], kind='zero',
                           bounds_error=False, fill_value=val[-1, 1])

    return interp_func(np.arange(0, total_duration, sampling_interval))


def calculate_velocity_from_updates(dy, dt, sampling_interval, total_duration,
                                    sigma=0.1):
    """Calculate animal velocity from dy/dt, as recorded from the rotary
    encoder '__position_updates' (for newer behavior files).

    Parameters
    ----------
    dy : array (n_samples,)
        Array of position updates, in centimeters
    dt : array (n_samples,)
        Array of time stamps for position updates, in seconds
    sampling_interval : float
        Time length of each bin in velocity trace
    total_duration : float
        Total length of trace, in seconds
    sigma : float, optional
        Standard deviation of the gaussian smoothing kernel to apply to the
        velocity trace, in seconds. Defaults to 0.1.

    Returns
    -------
    velocity : array (n_samples,)
    """

    bins = np.arange(0, total_duration + sampling_interval, sampling_interval)
    bin_idx = np.digitize(dt, bins=bins)

    # find sum displacements in each time bin
    dy_sums = [np.sum(dy[bin_idx == n]) for n in range(1, len(bins))]
    velocity = np.asarray(dy_sums) / sampling_interval  # convert to cm/sec

    # smooth
    sigma = int(np.ceil(sigma / sampling_interval))
    kernel = gaussian(8 * sigma, sigma)
    kernel /= kernel.sum()

    return convolve(velocity, kernel, mode='same')


def calculate_velocity(pos, sampling_interval, track_length,
                       sigma=0.1, max_discontinuity=0.2):
    """Calculate animal velocity from the change in position values across
    samples. Use this only if '__position_updates' is not available in the
    behavior dictionary.

    Parameters
    ----------
    pos : array (n_samples,)
        Array of position samples, scaled to range [0, 1]
    sampling_interval : float
        Time length of position bin
    track_length : float
        Length of track in millimeters
    sigma : float, optional
        Standard deviation of the gaussian smoothing kernel to apply to the
        velocity trace, in seconds. Defaults to 0.1.
    max_discontinuity : float (0, 1), optional
        Max allowable discontinuity in position from sample t to t + 1. If the
        position change is greater than this value, we assume a position reset
        occured (e.g. at the end of a lap, or ITI). The value of the violating
        bin is changed to match that of the preceding bin (i.e., assume a
        constant velocity from the prior sample during the reset). Defaults to
        0.2 (i.e. a fifth of the track length)

    Returns
    -------
    velocity : array (n_samples,)
    """

    delta_pos = np.concatenate([[0], np.diff(pos)])
    discont_idx = np.where(np.abs(delta_pos) > max_discontinuity)[0]

    # TODO : should we just interpolate?
    delta_pos[discont_idx] = delta_pos[discont_idx - 1]
    delta_pos *= (track_length / 10)  # change units to cm
    velocity = delta_pos / sampling_interval  # cm/sec

    # convert sigma to frames
    sigma = np.round(sigma / sampling_interval).astype('int')
    kernel = gaussian(sigma * 8, sigma)
    kernel /= kernel.sum()

    return convolve(velocity, kernel, mode='same')


def is_bruker(h5_path):
    try:
        get_prairie_xml_path(h5_path)
        return True
    except Exception:
        return False


def get_prairie_xml_path(h5_path):
    try:
        xml_name = [s.replace('.env', '.xml')
                    for s in os.listdir(os.path.split(h5_path)[0])
                    if s.endswith('.env')][0]
    except IndexError:
        xml_names = [s for s in os.listdir(os.path.split(h5_path)[0])
                     if s.endswith('.xml')]
        assert len(xml_names) == 1
        xml_name = xml_names[0] 

    xml_path = os.path.join(os.path.split(h5_path)[0], xml_name)
    if os.path.exists(xml_path):
        return xml_path
    else:
        raise Exception(
            "Could not locate Prairie xml/env files in h5 directory")


def get_prairie_frame_period(xml_path, round_result=True):
    """Parse the PrairieView xml to return the corrected frame period

    Parameters
    ----------
    xml_path : str
        Path to xml file
    round_result : bool, optional
        Whether to round the result to 6 decimal places. Defaults to True.
    """

    # TODO this is copied from the old experiment class and can definitely be
    # cleaned up considerably

    from xml.etree import ElementTree

    n_seqs = 2
    seqs = []
    _iter = ElementTree.iterparse(xml_path)
    while len(seqs) < n_seqs:
        try:
            _, elem = next(_iter)
        except StopIteration:
            break
        if elem.tag == 'Sequence':
            seqs.append(elem)

    if len(seqs) > 1:
        times = [float(s.find('Frame').get('absoluteTime'))
                 for s in seqs]
    else:
        times = [float(frame.get('absoluteTime'))
                 for frame in seqs[0].findall('Frame')[:n_seqs]]

    frame_period = np.diff(times).mean()
    if seqs[0].get('bidirectionalZ', 'False') == 'True':
        frame_period *= 2.0

    if round_result:
        return np.around(frame_period, 6)

    return frame_period


def is_scanimage(h5_path):
    raise NotImplementedError


def get_scanimage_frame_period():
    raise NotImplementedError


def is_femtonics(h5_path):
    raise NotImplementedError


def get_femtonics_frame_period():
    raise NotImplementedError


def update_h5_metadata(h5_path):
    """doc string"""

    h5 = h5py.File(h5_path, 'r+', libver='latest')

    if 'frame_period' not in h5['/imaging'].attrs.keys():

        if is_bruker(h5_path):
            xml_path = get_prairie_xml_path(h5_path)
            h5['/imaging'].attrs['frame_period'] = \
                get_prairie_frame_period(xml_path)
            h5['/imaging'].attrs['imaging_system'] = 'bruker'
        elif is_scanimage(h5_path):
            h5['/imaging'].attrs['frame_period'] = \
                get_scanimage_frame_period()
            h5['/imaging'].attrs['imaging_system'] = 'scanimage'
        elif is_femtonics(h5_path):
            h5['/imaging'].attrs['frame_period'] = \
                get_femtonics_frame_period()
            h5['/imaging'].attrs['imaging_system'] = 'femtonics'

    metadata = {k: v for k, v in h5['/imaging'].attrs.items()}
    h5.close()
    return metadata


def split_signals(expt, new_label, new_label_rois, current_label=None,
                  inverse=False, signal_type='*', channel='*', dry_run=False,
                  overwrite=False):
    """Split off a subset of ROIs to a new label, together with their signals.

    Parameters
    ----------
    expt : Experiment
        Experimet to get signals and rois from
    new_label : str
        New label to put new signals and ROIs in
    new_label_rois : array-like of ROI labels
        ROIs to move to new label. Only ROIs found within the current
        experiment will be included
    current_label : str
        Current label where the ROIs to be moved reside
    inverse : bool, optional
        Move all ROIs *not* in `new_label_rois` to new_label.
        Defaults to False.
    signal_type : str, optional
        The type of signal ({'raw', 'dfof', etc.}) to move. Defaults to all.
    channel : str, optional
        The channel ({'Ch1', 'Ch2', etc.}) to read signals from.
        Default to all.
    dry_run : bool, optional
        Show actions to be taken, without writing anything. Defaults to False.
    overwrite : bool, optional
        Overwrite key `new_label`, if it already exists in the signals file.
        Defaults to False.
    """
    print(f'In {expt.sima_path}:')
    rois = expt.roi_list(label=current_label)

    if inverse:
        new_rois = ROIList(
            [r for r in rois if r.label not in set(new_label_rois)])
    else:
        new_rois = ROIList([r for r in rois if r.label in set(new_label_rois)])

    roi_path = os.path.join(expt.sima_path, 'rois.pkl')

    with expt.signals_file() as signals_file:
        keys = signals_file.keys()

    for key in keys:
        _, ch_key, label_key, type_key = key.split('/')
        if (ch_key == channel or channel == '*') \
                and (type_key == signal_type or signal_type == '*') \
                and (label_key == current_label):

            signals = expt.signals(channel=ch_key, signal_type=type_key,
                                   label=current_label)

            if inverse:
                new_signals = signals.loc[
                    signals.index.difference(new_label_rois)]
            else:
                new_signals = signals.loc[
                    signals.index.intersection(new_label_rois)]

            # TODO: Handle this in a better way
            new_key = f'/{ch_key}/{new_label}/{type_key}'

            print(f"Splitting off {len(new_signals)} ROIs "
                  + f"from {key} -> {new_key}")
            with expt.signals_file(mode='a') as write_file:
                key_exists = new_key in write_file
                assert not key_exists or overwrite, \
                    f"Key {new_key} already exists in signal file. Set " \
                    + "overwrite=True to recalculate"
                if not dry_run:
                    write_file.put(new_key, new_signals)

    print(f"Splitting off {len(new_rois)} ROIs from "
          + f"{current_label} -> {new_label}")
    if not dry_run:
        new_rois.save(roi_path, label=new_label)

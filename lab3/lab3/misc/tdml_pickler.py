import json
from datetime import datetime
import numpy as np
import re
import os.path
import time
import pickle
from distutils.version import LooseVersion


# ------------------------------ MAIN FUNCTIONS ----------------------------- #


def find_files(directory, include_pickled=False):
    """Find tdml files in the specified directory

    Parameters
    ----------
    directory: str
        Path to search for tdml files
    include_pickled : bool, optional
        Whether to include a tdml file if a pickled version already exists at
        the same directory.

    Returns
    -------
    paths : list of str
        Paths for tdml files in directory
    """
    paths = []  # holds the paths
    pickled_paths = []  # temporally owns pickle files if they were found

    # checks if the path given isn't that of the tdml file
    if os.path.splitext(directory)[1] == '.tdml':
        paths.append(directory)

        # gets path of file without the .tdml
        path = os.path.splitext(directory)[0]

        # checks if there is .pkl file already present
        if os.path.exists('%s.pkl' % path):
            pickled_paths.append(path)  # stores the .pkl file

    # if we got a path for a directory
    else:
        # we walk through the directory and its sub-folders
        for dirpath, dirnames, filenames in os.walk(directory):
            # extends the paths array to hold all the .tdlm files' path
            paths.extend(
                map(lambda f: os.path.join(dirpath, f),
                    filter(lambda f: os.path.splitext(f)[1] == '.tdml',
                           filenames)))
            # extends the pickled_paths to hold all the .pkl file's path
            pickled_paths.extend(
                map(lambda f: os.path.join(dirpath, os.path.splitext(f)[0]),
                    filter(lambda f: os.path.splitext(f)[1] == '.pkl',
                           filenames)))

    if not include_pickled:
        # only return the paths that are not pickled
        paths = filter(
            lambda f: os.path.splitext(f)[0] not in pickled_paths, paths)

    return paths


def pickle_file(filepath):
    """Parses a tdml file and dumps information to pickle file

    Parameters
    ----------
    filepath : str
        Absolute or relative path to the file to parse
    """

    res = {}
    print(f'parsing {filepath}')

    last_ts = 0  # last time stamp

    with open(filepath, 'r') as f:
        for line in f:  # loops through lines
            try:
                data = json.loads(line)  # load json data of line
            except Exception:
                print(f'error parsing:\n {line}')

            last_ts = data.get('time', last_ts)  # gets the time stamp

            _time = data.get('time')
            parseKeys(
                globals(), data, _time, res, default, True,
                ['time', 'message'])

    if '__position_y' in res.keys():  # if position_y is in the key set
        res['__position_y'] = fixPosition(res['__position_y'],
                                          res['__position_updates'])

    if '__position_updates' in res.keys():
        res['__position_updates'] = \
                fillPositionTimes(res['__position_updates'])

    res['__trial_info']['laps'] = len(res.get('lap', []))

    # falling edge of the the 2nd sync pinc -- AOD bug: when acquiring
    # TSeries > 5min, acquisition has a 4s lag (vs ~500ms with <5min)
    if 'AOD_sync_pin' in res.keys():
        offset = res['AOD_sync_pin'][0][1]
        for key in filter(lambda k: k[0] != '_', res.keys()):
            if key not in ['trackLength', 'recordingDuration']:
                res[key] = res[key] - offset

    elif 'sync_pin' in res.keys():
        offset = res['sync_pin'][0][0]
        if np.isnan(offset):
            if np.isfinite(res['sync_pin'][0][1]):
                offset = res['sync_pin'][0][1] - 0.1
            else:
                offset = 0

        for key in filter(lambda k: k[0] != '_', res.keys()):
            if key not in ['trackLength', 'recordingDuration']:
                res[key] = res[key] - offset
    else:
        offset = 0

    if 'recordingDuration' not in res.keys():
        res['recordingDuration'] = last_ts - offset

    if "__position_y" in res.keys():
        res['__position_y'][:, 0] -= offset
        res['__position_y'] = res['__position_y'][np.where(
            res['__position_y'][:, 0] >= 0)]
        if res['__position_y'][0, 0] != 0:
            res['__position_y'] = np.insert(
                res['__position_y'], 0, [0, res['__position_y'][0, 1]], axis=0)
        treadmillPosition = np.array(res['__position_y'])
        treadmillPosition[:, 1] /= res['trackLength']
        treadmillPosition[:, 1] %= 1

        res['treadmillPosition'] = treadmillPosition

    if '__position_updates' in res.keys():
        res['__position_updates'][:, 0] -= offset
        res['__position_updates'] = res['__position_updates'][np.where(
            res['__position_updates'][:, 0] >= 0)]

    if 'stop_time' not in res['__trial_info'].keys():
        time_format = '%Y-%m-%d %H:%M:%S'
        start_ti = time.mktime(parseTime(res['__trial_info']['start_time']))
        res['__trial_info']['stop_time'] = datetime.fromtimestamp(
            start_ti + treadmillPosition[-1, 0]).strftime(time_format)

    output_file = f'{os.path.splitext(filepath)[0]}.pkl'
    print(f'writing {output_file}')
    with open(output_file, 'wb') as f:
        pickle.dump(res, f)

    print('finished writing')


def fix_NaNs_in_pkl(filename, save_result=True):
    """Attempts to fix NaNs present in the pickled data

    Parameters
    ----------
    filename : str
        Absolute or relative path to the file to fix
    save_results : bool, optional
        Flag that indicates whether to overwrite the pickled file
    """

    pkl_path = f'{os.path.splitext(filename)[0]}.pkl'
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    behavior_keys = filter(lambda k: re.match('^__.*', k) is None, data.keys())
    error_keys = [k for k in behavior_keys if np.any(np.isnan(data[k]))]
    context_info = data['__trial_info']['contexts']

    method = ''
    multiple_reward_flag = False
    for key in error_keys:
        cue_times = data[key]
        nan_times = np.where(np.isnan(cue_times))
        trial_length = data['__trial_info']['trial_length']
        track_length = float(data['__trial_info']['track_length'])

        if key in ['reward']:
            if 'reward' not in data['__reward_contexts']:
                multiple_reward_flag = True
                continue
            else:
                method = 'position'
        elif key in ['licking', 'water', 'tone']:
            method = 'time'
        elif re.match('.*_pin$', key) is not None:
            method = 'time'

        if key in context_info.keys():
            if context_info[key].get('locations') is not None:
                if isinstance(context_info[key]['locations'], list):
                    method = 'position'
                else:
                    method = 'position_shuffled'
            else:
                print('skipping %s' % key)
                continue
        print(f'{key} - {method}')
        if method == 'time':
            if key == 'sync_pin':
                cue_length = 0.1
            else:
                cue_length = np.nanmean(np.diff(cue_times))
                if cue_length == np.nan:
                    raise Exception("Unable to Determine Cue Length")
            for idx in nan_times[0][nan_times[1] == 1]:
                if cue_times[idx, 0] + cue_length < trial_length:
                    cue_times[idx, 1] = cue_times[idx, 0] + cue_length

            for idx in nan_times[0][nan_times[1] == 0]:
                if cue_times[idx, 1] - cue_length > 0:
                    cue_times[idx, 0] = cue_times[idx, 1] - cue_length

        elif method == 'position':
            position = data['treadmillPosition']
            radius = context_info[key]['radius'] / track_length
            max_duration = context_info[key].get('max_duration')
            for idx, j in zip(nan_times[0], nan_times[1]):
                if j == 1:
                    if max_duration is not None and \
                            cue_times[idx, 0] < max_duration:
                        continue
                    _position = position[position[:, 0] > cue_times[idx, 0]]
                    if max_duration is not None:
                        _position = _position[_position[:, 0]
                                              < _position[0, 0] + max_duration]
                else:
                    position_rev = position[::-1]
                    _position = position_rev[position_rev[:, 0]
                                             < cue_times[idx, 1]]
                    if max_duration is not None:
                        _position = _position[_position[:, 0]
                                              > _position[0, 0] - max_duration]

                if _position.shape[0] < 2:
                    continue

                location = None
                for location in context_info[key]['locations']:
                    location = location / track_length
                    if location - radius <= _position[0, 1] \
                            <= location + radius:
                        break
                if location is None:
                    raise Exception("start position not in context location")

                _position = _position[np.logical_not(np.logical_and(
                    location - radius <= _position[:, 1],
                    _position[:, 1] <= location + radius))]
                if _position.shape[0] == 0:
                    continue
                else:
                    cue_times[idx, j] = _position[0, 0]

        elif method == 'position_shuffled':
            position = data['treadmillPosition']
            radius = context_info[key]['radius'] / track_length
            max_duration = context_info[key].get('max_duration')
            for idx, j in zip(nan_times[0], nan_times[1]):
                if j == 1:
                    if max_duration is not None and \
                            cue_times[idx, 0] < max_duration:
                        continue
                    _position = position[position[:, 0] > cue_times[idx, 0]]
                    if max_duration is not None:
                        _position[_position[:, 0]
                                  < _position[0, 0] + max_duration]
                else:
                    position_rev = position[::-1]
                    _position = position_rev[position_rev[:, 0]
                                             < cue_times[idx, 1]]
                    if max_duration is not None:
                        _position[_position[:, 0]
                                  > _position[0, 0] - max_duration]

                if _position.shape[0] < 2:
                    continue

                if j == 1 and _position[1, 1] > _position[0, 1] or \
                        j == 0 and _position[1, 1] < _position[0, 1]:
                    location = _position[0, 1] + radius
                else:
                    location = _position[0, 1] - radius

                _position = _position[np.logical_not(np.logical_and(
                    location - radius <= _position[:, 1],
                    _position[:, 1] <= location + radius))]
                if _position.shape[0] == 0:
                    continue
                else:
                    cue_times[idx, j] = _position[0, 0]

    if multiple_reward_flag:
        all_reward_times = [data[k] for k in data['__reward_contexts']]
        all_reward_times = np.vstack(all_reward_times)
        data['reward'] = all_reward_times[all_reward_times[:, 0].argsort()]

    with open(pkl_path, 'r') as f:
        behavior_data = pickle.load(f)

    print('\nchanges to be made ...')
    for cue in error_keys:
        print(f'cue: {cue}')
        for idx in np.where(behavior_data[cue] != data[cue])[0]:
            print(f'{str(behavior_data[cue][idx])} -> {str(data[cue][idx])}')
        behavior_data[cue] = data[cue]

    if save_result:
        print('saving ...')
        with open(pkl_path, 'w') as f:
            pickle.dump(behavior_data, f)
    print('\n\n')


# -------------------------- TDML PARSING FUNCTIONS ------------------------- #


def parseTime(time_string):
    """Attempts to parse the given time string with an appropriate, readable
    format

    Parameters
    ----------
    time_string: str
        String to parse

    Returns
    -------
    time : parsed time
    """

    # acceptable time formats
    time_formats = ['%Y-%m-%d %H:%M:%S', '%d-%b-%Y %H:%M:%S',
                    '%Y-%m-%d%H:%M:%S', '%Y-%m-%d-%Hh%Mm%Ss']

    # loop through all the time formats
    for frmt in time_formats:
        try:  # attempt to parse the time, if successful return
            return time.strptime(time_string, frmt)
        except ValueError:  # otherwise pass
            pass


def parseKeys(functions, msg, _time, _dict, default=None, filtered=False,
              to_filter=None):
    """Attempts to call all the functions present in the message being passed.

    Parameters
    ----------
    :param functions: dictionary of functions present in the project
    :param msg: Json retrieved data (usally a dict)
    :param _time: time of data performed
    :param _dict: stores current trial's formatted info
    :param default: function to call if a key in message is not a previously
    defined function
    :param filtered: boolean that indicates if the message should be filtered
    :param to_filter: array of keys to filter
    :return: Void function
    """

    keys = msg.keys()  # get the keys from the message

    if filtered:  # check if we require some kind of filter
        if to_filter:  # check that we have the filter array
            # filter the list given the parameters
            keys = list(filter(lambda k: k not in to_filter, keys))

    for key in keys:  # loop through the keys
        # check if the function exists
        if '_{}_'.format(key) in functions.keys():
            # if it exists, call the function
            functions['_{}_'.format(key)](msg[key], _time, _dict)
        # else check that there is a default method to call
        elif default is not None:
            default(key, msg[key], _time, _dict)  # call default method


def default(key, value, _time, _dict):
    """
    Default method call for all parseKey processes
    :param key: if is to be edit
    :param value: if is to be edit
    :param _time: time of trial
    :param _dict: stores current trial's formatted info
    :type key: str
    :type value: str, int, bool, dict, etc ...
    :type _time: str
    :type _dict: dict
    :return: None
    :rtype: void
    """
    insert(_dict, '__trial_info', {str(key): value})


def insert(_dict, key, value):
    """
    Safe inserts a key in the dictionary
    :param _dict: stores current trial's formatted info
    :param key: to insert
    :param value: of the key
    :return: Void method
    """
    # attempt to get the key, if doesn't exists, it create an empty dictionary
    _dict[key] = _dict.get(key, {})
    _dict[key].update(value)  # update the key value with the intended value


def fillPositionTimes(position_updates):
    """
    populates missing times for the position_updates array
    :param position_updates: holds position time changes
    :type position_updates: np.array
    :return: updated position_updates
    :rtype: np.array
    """
    position_updates = np.flip(position_updates, axis=0)
    missingTimes = np.where([p is None for p in position_updates])[0]

    for t in missingTimes:
        position_updates[time, 0] = position_updates[t - 1, 0] \
                                    - position_updates[t, 2] * 10e-3
    return np.flip(position_updates, axis=0).astype(float)


def fixPosition(position_array, position_updates):
    """
    fixes the missing entries in the position_updates list
    :param position_array: position data
    :param position_updates: position time changes
    :type position_array: np.array
    :type position_updates: np.array
    :return: updated position data
    :rtype: np.array
    """
    position_updates = position_updates.astype(float)
    nans = np.where(np.isnan(position_updates))[0]
    for idx in np.flip(nans, 0):
        position_updates[idx + 1][1] += position_updates[idx][1]
        position_updates = np.delete(position_updates, idx, axis=0)

    delta = position_updates[1:, 0] - position_updates[1:, 2] * 10e-3
    new_times = delta[position_updates[:-1, 0] < delta]

    for i, t in enumerate(np.flip(new_times, axis=0)):
        idx = np.where(position_array[:, 0] > t)[0][0]
        position_array = np.insert(
            position_array, idx, [new_times[-(i + 1)], position_array[idx, 1]],
            axis=0)

    return position_array


def _version_(ver, _time, _dict):
    """
    checks trial's version comparability, and stores in information
    :param ver: version of current file
    :param _time: UNUSED
    :param _dict: stores current trial's formatted info
    :type ver: str
    :type _time: str
    :type _dict: dict
    :return: None
    :rtype: void
    """
    if LooseVersion(ver) < LooseVersion('0.0.6b5'):
        raise Exception('unsupported behaviorMate version')
    insert(_dict, '__trial_info', {'version': ver})


def _start_(time_str, _time, _dict):
    """
    adds the start time for the current trial to the reformatted data
    :param time_str: time to be parsed
    :param _time:  UNUSED
    :param _dict: stores current trial's formatted info
    :type time_str: str
    :type _time: str
    :type _dict: dict
    :return: None
    :rtype: void
    """
    sql_time_format = '%Y-%m-%d %H:%M:%S'
    insert(_dict, '__trial_info',
           {'start_time':
            time.strftime('%Y-%m-%d-%Hh%Mm%Ss',
                          time.strptime(time_str, sql_time_format))})


def _stop_(time_str, _time, _dict):
    """
    adds a stop time flag to the reformatted trial data
   :param time_str: time to be parsed
    :param _time:  UNUSED
    :param _dict: stores current trial's formatted info
    :type time_str: str
    :type _time: str
    :type _dict: dict
    :return: None
    :rtype: void
    """
    sql_time_format = '%Y-%m-%d %H:%M:%S'  # time format
    insert(_dict, '__trial_info',
           {'stop_time': time.strftime(
                    '%Y-%m-%d-%Hh%Mm%Ss',
                    time.strptime(time_str, sql_time_format))})
    if _dict.get('__trial_info', {}).get('trial_length') is None:
        insert(_dict, '__trial_info', {'trial_length': _time})


def _context_(msg, _time, _dict):
    """
    Ignores the context key word at a global level
    :param msg: ignored
    :param _time: ignored
    :param _dict: ignored
    :return:
    """
    pass


def _comments_(comments, _time, _dict):
    """
    Reformat comments for the current trial
    :param comments: comments to insert
    :param _time: UNUSED
    :param _dict: stores current trial's formatted info
    :type comments: str
    :type _time: str
    :type _dict: dict
    :return:
    """
    insert(_dict, '__trial_info', {'comments': comments})


def _settings_(msg, _time, _dict):
    """
    reformat settings information for the current trial
    :param msg:  dictionary with the trial specification
    :param _time: time of trial
    :param _dict: stores current trial's formatted info
    :type msg: dict
    :type _time: str
    :type _dict: dict
    :return: None
    :rtype: void
    """

    def _sync_pin_(msg, _time, _dict):  # adds the sync pin information
        insert(_dict, '__valve_ids', {msg: 'sync_pin'})

    def _track_length_(msg, _time, _dict):  # adds the track_length information
        _dict['trackLength'] = msg
        insert(_dict, '__trial_info', {'track_length': msg})

    def _trial_length_(msg, _time, _dict):  # adds the trial length information
        _dict['recordingDuration'] = msg
        insert(_dict, '__trial_info', {'trial_length': msg})

    # adds the general setting information for each context to the dictionary
    def _contexts_(msg, _time, _dict):
        contexts = {c['id']: c for c in msg}  # get's the name of each context
        for context in contexts.keys():  # traverse the whole list
            for valve in contexts[context].get('valves', []):
                # store the valve information separately
                insert(_dict, '__valve_ids', {valve: str('%s_pin' % context)})

        # add the context information to trial_info
        insert(_dict, '__trial_info', {'contexts': contexts})

    def _reward_(msg, _time, _dict):  # adds reward data to the file
        insert(_dict, '__trial_info', {'reward': msg})

    parseKeys(locals(), msg, _time, _dict, default=default)

    contexts = _dict['__trial_info'].get('contexts', {})

    reward_contexts = list(filter(lambda n: 'reward' in n, contexts.keys()))

    if 'reward' in _dict['__trial_info']:
        reward = _dict['__trial_info']['reward']
        if reward.get('id', 'reward') not in reward_contexts:
            reward_contexts.append(reward.get('id', 'reward'))

    reward_pins = []

    for context in reward_contexts:
        reward_pins.extend(contexts[context].get('valves'))

    if 'reward' in _dict['__trial_info']:
        if reward.get('pin') is not None and \
                reward.get('pin') not in reward_pins:
            reward_pins.append(reward['pin'])

    _dict['__reward_contexts'] = reward_contexts
    _dict['__reward_pins'] = reward_pins


def _behavior_mate_(msg, _time, _dict):
    """
    reformat messages posted by behavior_mate pc for the current trial
    :param msg:  dictionary with the position specifications
    :param _time: time of trial
    :param _dict: stores current trial's formatted info
    :type msg: dict
    :type _time: str
    :type _dict: dict
    :return: None
    :rtype: void
    """

    def _context_(msg, _time, _dict):
        if 'id' not in msg.keys():
            return

        _id = msg['id']
        if msg['action'] == 'start':
            if _dict.get(_id, np.zeros((1, 2)))[-1, 1] != np.NaN:
                _dict[_id] = np.vstack((
                    _dict.get(_id, np.empty((0, 2))), [_time, np.NaN]))

        elif msg['action'] == 'stop':
            if np.isnan(_dict.get(_id, np.zeros((1, 2)))[-1, 1]):
                _dict[_id][-1, 1] = _time

    if 'scene' in msg.keys():
        parseScene(msg, _dict)

    for key in msg.keys():
        if '_{}_'.format(key) in locals().keys():
            locals()['_{}_'.format(key)](msg[key], _time, _dict)


def _position_controller_(msg, _time, _dict):
    """
    reformat the position information for the current trial
    :param msg:  dictionary with the position specifications
    :param _time: time of trial
    :param _dict: stores current trial's formatted info
    :type msg: dict
    :type _time: str
    :type _dict: dict
    :return: None
    :rtype: void
    """

    def _lap_reset_(msg, _time, _dict, _millis):
        if 'position_lap_reader' not in _dict.keys():
            _dict['position_lap_reader'] = np.array([])

        _dict['position_lap_reader'] = np.append(
            _dict['position_lap_reader'], _time)

    def _position_(msg, _time, _dict, _millis):
        if '__position_updates' not in _dict.keys():
            _dict['__position_updates'] = np.array([]).reshape((0, 4))
        _dict['__position_updates'] = np.vstack(
            (_dict['__position_updates'], [_time, msg['dy'], msg['dt'],
                                           _millis]))

    for key in msg.keys():
        if '_{}_'.format(key) in locals().keys():
            locals()['_{}_'.format(key)](msg[key], _time, _dict,
                                         msg.get('millis'))


def _y_(msg, _time, _dict):
    """
    reformat the y component for the current trial
    :param msg:  dictionary with the trial specification
    :param _time: time of trial
    :param _dict: stores current trial's formatted info
    :type msg: dict
    :type _time: str
    :type _dict: dict
    :return: None
    :rtype: void
    """
    if '__position_y' not in _dict.keys():
        _dict['__position_y'] = np.array([]).reshape((0, 2))

    _dict['__position_y'] = np.vstack(
        (_dict['__position_y'], [_time, float(msg)]))


def _lap_(msg, _time, _dict):
    """
    reformat lap information
    :param msg:  dictionary with the lap specifications
    :param _time: time of trial
    :param _dict: stores current trial's formatted info
    :type msg: dict
    :type _time: str
    :type _dict: dict
    :return: None
    :rtype: void
    """
    if 'lap' not in _dict.keys():
        _dict['lap'] = np.array([])

    _dict['lap'] = np.append(_dict['lap'], _time)


def _behavior_controller_(msg, _time, _dict):
    """
    reformat the behavior information for the current trial
    :param msg:  dictionary with the position specifications
    :param _time: time of trial
    :param _dict: stores current trial's formatted info
    :type msg: dict
    :type _time: str
    :type _dict: dict
    :return: None
    :rtype: void
    """

    def context_update(this_id, this_msg, _time, _dict):
        if this_id not in _dict.keys():
            _dict[this_id] = np.array([]).reshape((0, 2))
        if this_msg.get('action') == 'start' or \
                this_msg.get('action') == 'open':
            _dict[this_id] = np.vstack((_dict[this_id], [_time, np.NaN]))
        elif this_msg.get('action') == 'stop' \
                or this_msg.get('action') == 'close':
            if _dict[this_id].shape[0] == 0 or not \
                    np.isnan(_dict[this_id][-1][-1]):
                _dict[this_id] = np.vstack(
                    (_dict[this_id], [np.NaN, _time]))
            else:
                _dict[this_id][-1][-1] = _time

    def _context_(msg, _time, _dict):
        context_id = str(msg['id'])  # get id from context
        if context_id in _dict.get('__valve_ids', {}).values() or \
                context_id == 'water':  #
            context_id = str(context_id) + '_'

        context_update(context_id, msg, _time, _dict)

        if context_id != 'reward' and \
                context_id in _dict.get('__reward_contexts'):
            msg['id'] = 'reward'
            _context_(msg, _time, _dict)

    def _valve_(msg, _time, _dict):
        valve_id = _dict.get('__valve_ids', {}).get(msg['pin'])

        if valve_id is None:
            valve_id = 'pin_%i' % msg['pin']

        context_update(valve_id, msg, _time, _dict)

        if msg['pin'] in _dict['__reward_pins']:
            context_update('water', msg, _time, _dict)

    def _tone_(msg, _time, _dict):
        valve_id = _dict.get('__tone_ids', {}).get(msg['pin'])

        if valve_id is None:
            valve_id = 'tone_%i' % msg['pin']

        context_update(valve_id, msg, _time, _dict)

    def _lick_(msg, _time, _dict):
        context_update('licking', msg, _time, _dict)

    parseKeys(locals(), msg, _time, _dict)


def parseScene(data, _dict):
    scene = data.get('id', data['scene'])
    if data['context']['action'] == 'start':
        if _dict.get(scene, np.zeros((1, 2)))[-1, 1] != np.NaN:
            _dict[scene] = np.vstack((
                _dict.get(scene, np.empty((0, 2))), [data['time'], np.NaN]))

    elif data['context']['action'] == 'stop':
        if np.isnan(_dict.get(scene, np.zeros((1, 2)))[-1, 1]):
            _dict[scene][-1, 1] = data['time']


# for populating the documentation website
__all__ = [
    'find_files',
    'pickle_file',
    'fix_NaNs_in_pkl'
]

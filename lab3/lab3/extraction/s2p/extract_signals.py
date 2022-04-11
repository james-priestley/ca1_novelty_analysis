import os
import shutil
import itertools as it
import multiprocessing

import numpy as np
from suite2p.run_s2p import run_s2p
try:
    from suite2p.utils import init_ops
except ModuleNotFoundError:
    from suite2p.io.utils import init_ops
from sima.sequence import _fill_gaps

from lab3.misc.progressbar import ProgressBar
from lab3.misc.pool_helpers import (bind_locked_get_frame, pool_helper,
                                    slice_sequence)
from lab3.misc.sima_sequences import _NumPyFuncSequence


sequence_store = 0
lock = 0


def dump_to_binary(ds, signal_channel='Ch2', static_channel=None,
                   fill_gaps=False):
    """Converts SIMA imaging dataset to binary file, in the format expected by
    Suite2p. Along the way, additional 'ops' dictionaries are created for
    each imaging plane and saved to disk in the suite2p directory.

    If the dataset has multiple sequences, they will be concatenated in the
    binary file as if they were one continuous recording. For this reason,
    all sequences should be of compatible frame shape (zyxc). This allows batch
    processing of datasets from the same FOV, but special consideration should
    be given for motion correction. If the individual datasets have already
    been motion corrected, you must align the sequences and crop them
    accordingly when constructing the batch imaging dataset object.
    This step is unnecessary if using Suite2p's motion correction, as the
    frames of all sequences will be aligned to a common reference.

    Parameters
    ----------
    ds : Suite2pStrategy (sima.imaging.ImagingDataset)
        A Suite2p strategy instance (e.g. Suite2pImagingDataset), initialized
        from the sima folder with the desired imaging dataset. All sequences in
        the imaging dataset will be concatenated and written to the binary
        file.
    signal_channel : str or int
        Channel with dynamic signal (e.g. GCaMP). Defaults to 'Ch2'.
    static_channel : str or int
        Channel with static marker (e.g. tdTomato). If passed, the static
        channel is also dumped to a binary file. Defaults to None.
    fill_gaps : bool
        Fill in NaNed pixels with data from adjacent frames. If False, any
        NaNed pixels will appear as zeros in the binary file. Defaults to
        False.

    Returns
    -------
    ops_list : list of dicts
        List of ops dictionaries (one for each imaging plane). If there is only
        one plane, or collapse_z=True, it will be a list of one dict.

    Notes
    -----
    The format of the binary file and the addtiional parameters added to the
    ops dictionaries has not changed from 0.5.5 to the latest versions as far
    as we are aware, so this function should generalize.
    """

    # create list of channels to dump to binary
    channel_list = [ds._resolve_channel(signal_channel)]
    if static_channel is not None:
        channel_list.append(ds._resolve_channel(static_channel))
        reg_file_chan2, meanImg_chan2 = [], []

    Ly, Lx = ds.frame_shape[1:3]

    reg_file, meanImg = [], []
    ops_list = init_ops({**ds.ops, **ds.db})

    for pidx, ops in enumerate(ops_list):

        # open binary for writing
        reg_file.append(open(ops['reg_file'], 'wb'))
        meanImg.append(np.zeros((Ly, Lx)))

        if static_channel is not None:
            reg_file_chan2.append(open(ops['reg_file_chan2'], 'wb'))
            meanImg_chan2.append(np.zeros((Ly, Lx)))

    # note that if not fill_gaps, nans will be set to zero in the binary
    save_frames = [_fill_gaps(iter(seq), iter(seq))
                   if fill_gaps else iter(seq)
                   for seq in ds.sequences]

    # iterate over sequence and dump to binaries
    # TODO clean this up
    frame_counter = 0
    for sidx, seq in enumerate(save_frames):
        p = ProgressBar(ds.sequences[sidx].shape[0])

        for fidx, frame in enumerate(seq):
            frame_counter += 1
            p.update(fidx)

            for pidx, plane in enumerate(frame):
                for ch_idx, channel in enumerate(channel_list):
                    if ch_idx == 0:
                        reg_file[pidx].write(bytearray(
                            plane[:, :, channel].astype('uint16')))
                        meanImg[pidx] += plane[:, :, channel]
                    else:
                        reg_file_chan2[pidx].write(bytearray(
                            plane[:, :, channel].astype('uint16')))
                        meanImg_chan2[pidx] += plane[:, :, channel]
        p.end()

    # save plane options, close binary files
    for pidx, ops in enumerate(ops_list):
        ops['Ly'] = Ly
        ops['Lx'] = Lx
        ops['nframes'] = frame_counter
        ops['meanImg'] = meanImg[pidx] / frame_counter
        if not ops['do_registration']:
            ops['refImg'] = ops['meanImg']
        reg_file[pidx].close()

        if static_channel is not None:
            ops['meanImg_chan2'] = meanImg_chan2[pidx] / frame_counter
            reg_file_chan2[pidx].close()

        np.save(ops['ops_path'], ops)

    # save ops_list
    np.save(ds.ops_list_path, ops_list)
    return ops_list


def _write_data(sidx, paths, fill_gaps, signal_channel, static_channel):
    """ Method for performing parallel dump of imaging data into binary files.

    Parameters
    ----------
    sidx : int
        the index of the sequence in the global sequence store to write.
    paths : list of str
        the folder path to store temporary data in for each of the reg_files.
    fill_gaps : bool
        Fill in NaNed pixels with data from adjacent frames. If False, any
        NaNed pixels will appear as zeros in the binary file. Defaults to
        False.
    signal_channel : int
        Channel index with dynamic signal (e.g. GCaMP).
    static_channel : int
        Channel index with static marker (e.g. tdTomato). If passed, the static
        channel is also dumped to a binary file.

    Returns
    -------
    tmp_file_paths : list of str
        list of temporary files, one per plane of imaging data,  storing the
        imaging data converted to the binary format stored.
    meanImg : numpy.array
        the sum of all of the imaging frames exported, used in calculating the
        mean image for ths dataset.
    """
    global sequence_store
    global lock

    seq = sequence_store[sidx]
    tmp_filepaths = [os.path.join(path, 'data_%s.bin' % sidx)
                     for path in paths]
    tmp_files = list(map(lambda p: open(p, 'wb'), tmp_filepaths))
    if static_channel is not None:
        tmp_filepaths_chan2 = [os.path.join(path, 'data_chan2_%s.bin' % sidx)
                               for path in paths]
        tmp_files_chan2 = list(map(lambda p: open(p, 'wb'),
                                   tmp_filepaths_chan2))

    meanImg = np.zeros(seq.shape[1:])

    # note that if not fill_gaps, nans will be set to zero in the binary
    seq_iter = _fill_gaps(iter(seq), iter(seq)) if fill_gaps else iter(seq)

    p = ProgressBar(seq.shape[0])
    for fidx, frame in enumerate(seq_iter):
        p.update(fidx)
        meanImg += frame
        for pidx, plane in enumerate(frame):
            tmp_files[pidx].write(
                bytearray(plane[..., signal_channel].astype('int16')))
            if static_channel is not None:
                tmp_files_chan2[pidx].write(
                    bytearray(plane[..., static_channel].astype('int16')))

    list(map(lambda f: f.close(), tmp_files))

    if static_channel is not None:
        list(map(lambda f: f.close(), tmp_files_chan2))
        tmp_filepaths = [tmp_filepaths, tmp_filepaths_chan2]
    else:
        tmp_filepaths = [tmp_filepaths]

    p.end()
    return (tmp_filepaths, meanImg)


def _append_files(tmp_files, reg_files, sidx):
    """ Concatenates the data in binary files. Usefull when performing parallel
    dump to binary.

    Parameters
    ----------
    tmp_files : list of list of str
        Temporary files to be combined. Files will be deleted after being
        dumped into destination file. Outer list is for each plane in the
        dataset. Inner list is each file to be combined.
    reg_files : list of str
        Resulting binary file to combine tmp_files into.
    sidx : int
        sequence index being saved. if sidx is zero then new reg_files are
        created, otherwise the data is appended to the existing reg_files
    """
    print('appending_files')
    for pidx, reg_filepath in enumerate(reg_files):
        if sidx == 0:
            shutil.move(tmp_files[pidx].pop(0), reg_filepath)
        with open(reg_filepath, 'ab') as f:
            for tmp_file in tmp_files[pidx]:
                with open(tmp_file, 'rb') as f2:
                    while True:
                        _bytes = f2.read(1024)
                        if not _bytes:
                            break
                        f.write(_bytes)
                print('removing %s' % tmp_file)
                os.remove(tmp_file)


def parallel_dump_to_binary(ds, signal_channel='Ch2', static_channel=None,
                            fill_gaps=False, n_processes=1):
    """Converts SIMA imaging dataset to binary file, in the format expected by
    Suite2p. Along the way, additional 'ops' dictionaries are created for
    each imaging plane and saved to disk in the suite2p directory.

    see the method :meth:`lab3.extraction.s2p_helpers.dump_to_binary`

    Parameters
    ----------
    ds : Suite2pStrategy (sima.imaging.ImagingDataset)
        A Suite2p strategy instance (e.g. Suite2pImagingDataset), initialized
        from the sima folder with the desired imaging dataset. All sequences in
        the imaging dataset will be concatenated and written to the binary
        file.
    signal_channel : str or int
        Channel with dynamic signal (e.g. GCaMP). Defaults to 'Ch2'.
    static_channel : str or int
        Channel with static marker (e.g. tdTomato). If passed, the static
        channel is also dumped to a binary file. Defaults to None.
    fill_gaps : bool
        Fill in NaNed pixels with data from adjacent frames. If False, any
        NaNed pixels will appear as zeros in the binary file. Defaults to
        False.

    Returns
    -------
    ops_list : list of dicts
        List of ops dictionaries (one for each imaging plane). If there is only
        one plane, or collapse_z=True, it will be a list of one dict.
    """

    global lock
    global sequence_store
    lock = multiprocessing.Lock()

    signal_channel = ds._resolve_channel(signal_channel)
    if static_channel is not None:
        static_channel = ds._resolve_channel(static_channel)

    sequences = ds.sequences
    num_planes, Ly, Lx = sequences[0].shape[1:4]
    ops_list = init_ops({**ds.ops, **ds.db})
    for pidx in range(num_planes):

        if not os.path.isdir(ops_list[pidx]['save_path']):
            os.makedirs(ops_list[pidx]['save_path'])

    # iterate over sequence and dump to binaries
    for sidx, seq in enumerate(sequences):
        if n_processes > 1:
            bind_locked_get_frame(seq, lock)
            sequence_store = list(slice_sequence(seq, n_processes=n_processes))
            pool = multiprocessing.Pool(processes=n_processes)

        else:
            pool = None
            sequence_store = [seq]

        results = pool_helper(
            pool,
            _write_data,
            range(len(sequence_store)),
            it.repeat([o['save_path'] for o in ops_list]),
            it.repeat(fill_gaps), it.repeat(signal_channel),
            it.repeat(static_channel)
        )

        if pool is not None:
            pool.close()
            pool.join()
        else:
            results = list(results)

        weights = [s.shape[0] for s in sequence_store]
        sequence_store = None
        meanImages = np.array([r[1] for r in results])
        meanImages = meanImages / np.array(weights)[:, None, None, None, None]

        tmp_files = np.array([r[0] for r in results])
        tmp_file_list = tmp_files[:, 0, :].T.tolist()
        _append_files(tmp_file_list, [o['reg_file'] for o in ops_list], sidx)
        if static_channel is not None:
            tmp_files_chan2 = tmp_files[:, 1, :].T.tolist()
            _append_files(
                tmp_files_chan2, [o['reg_file_chan2'] for o in ops_list], sidx)

    # save plane options
    for pidx, ops in enumerate(ops_list):
        ops['Ly'] = Ly
        ops['Lx'] = Lx
        ops['nframes'] = seq.shape[0]
        ops['meanImg'] = meanImages[pidx, ..., signal_channel]
        if not ops['do_registration']:
            ops['refImg'] = ops['meanImg']

        if static_channel is not None:
            ops['meanImg_chan2'] = meanImages[pidx, ..., static_channel]

        np.save(ops['ops_path'], ops)

    np.save(ds.ops_list_path, ops_list)

    return ops_list


def accept_all_rois(save_dir):
    """ Places all ROIs in a suite2p dataset in the "is cell" category.

    Parameters
    ----------
    save_dir : str
        File path to walk. All iscell.npy files contianed in this directory or
        in a subdirectory of save_dir will be edited to accept all the ROIs.
    """

    for root, dirs, files in os.walk(save_dir):
        if 'iscell.npy' in files:
            cells = np.load(os.path.join(root, 'iscell.npy'))
            cells[:, 0] = 1
            np.save(os.path.join(root, 'iscell.npy'), cells)


def find_masked_frames(seq):
    """Find indices of masked frames in sequence"""
    masked_frames = []
    current_seq = seq
    while True:
        try:
            masked_frames += current_seq._mask_dict.keys()
        except AttributeError:
            pass
        try:
            current_seq = current_seq._base
        except AttributeError:
            break

    return np.unique(masked_frames)


def extract(ds, signal_channel='Ch2', static_channel=None, fs=10,
            collapse_z=True, fill_gaps=None, remove_masked_frames=True,
            register=False, sparse_mode=True, spatial_scale=0, diameter=None,
            overwrite=False, reprocess=False, n_processes=1, ops_kws={}):
    """
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
    sparse_mode : bool, optional
        Whether to use Suite2p's sparse mode. Defaults to True
    spatial_scale : int {0, 1, 2, 3}, optional
        ROI scale parameter for sparse model. 0 = multiscale, 1 = 6 pixels,
        2 = 12 pixels, 3 = 24 pixels, 4 = 48 pixels. Choose the closest value.
        Defaults to 0.
    diameter : int, optional
        Expected diameter of cells, in pixels. Defaults to 15. Ignored
        if sparse_mode is True.
    overwrite : bool, optional
        If True, existing Suite2p folder is deleted and dataset is
        reprocessed from the beginning of the pipeline. Defaults to False.
    reprocess : bool, optional
        If True, rerun Suite2p analysis on existing binary file.
        This can be used for example to extract ROIs multiple times with
        different parameters, without repeating the binary conversion and
        registration each time. Reprocess and overwrite cannot both be True.
        Not tested - we may need to rewrite the ops dictionaries still.
    ops_kws : dict, optional
        Pass additional keyword-argument pairs as a dictionary to
        override other Suite2p default settings.
    """

    if collapse_z and ds.frame_shape[0] > 1:
        ds.sequences = \
            [_NumPyFuncSequence(s, 'max', axis=0) for s in ds.sequences]

    # set user parameters and create ops dictionary
    ops_kws['nchannels'] = 2 if static_channel is not None else 1
    ops_kws['fs'] = fs
    ops_kws['do_registration'] = 1 if register else 0
    ops_kws['nplanes'] = ds.frame_shape[0]
    ops_kws['sparse_mode'] = sparse_mode
    if sparse_mode:
        ops_kws['spatial_scale'] = spatial_scale
    else:
        if not diameter:
            raise ValueError("If sparse_mode is False, you must specify a "
                             + "cell diameter")
        else:
            ops_kws['diameter'] = diameter
    ds.ops = ops_kws  # any unspecified parameters take their default values

    if not reprocess:
        # convert imaging data into suite2p binary file and create/save
        # additional settings files
        if fill_gaps is None:
            fill_gaps = not register

        print("Converting imaging data to Suite2p binary file\n"
              + f"(concatenating {len(ds.sequences)} sequences)")
        if n_processes == 1:
            dump_to_binary(ds, signal_channel=signal_channel,
                           static_channel=static_channel,
                           fill_gaps=fill_gaps)
        else:
            parallel_dump_to_binary(
                ds, signal_channel=signal_channel, fill_gaps=fill_gaps,
                static_channel=static_channel, n_processes=n_processes)

    else:
        print("Re-running Suite2p on existing binary file, if possible")

        # update ops files with new parameters
        new_ops_list = ds.ops_list
        for idx, ops in enumerate(new_ops_list):
            ops = {**ops, **ds.ops, **ds.db}
            np.save(ops['save_path'], ops)
            new_ops_list[idx] = ops
        np.save(ds.ops_list_path, new_ops_list)

    # handle masked frames
    try:
        os.unlink(ds.bad_frames_path)
    except FileNotFoundError:
        pass
    if remove_masked_frames:
        bad_frames = []
        frame_counter = 0
        for seq in ds.sequences:
            bad_frames.append(find_masked_frames(seq) + frame_counter)
            frame_counter += len(seq)
        bad_frames = np.concatenate(bad_frames)

        if len(bad_frames):
            # save in npy file, which will be recognized by suite2p
            np.save(ds.bad_frames_path, bad_frames)
            print(f"Ignoring frames:\n{bad_frames}")
        else:
            print("No frames to mask")

    # run suite2p main function
    print("\n\n-------ENTERING SUITE2P-------\n\n")
    run_s2p(ds.ops, ds.db)

# def extract_with_055(self, signal_channel='Ch2', static_channel=None,
#                      collapse_z=True, fill_gaps=None,
#                      remove_masked_frames=True, register=False,
#                      x_size=10, y_size=10, overwrite=False, reprocess=False,
#                      ops_kws={}):
#     """Provides backward compatibility for extraction with Suite2p 0.5.5"""
#
#     # we can reuse the extract function I think. Just override some other
#     # arguments. we need to set sparse_mode to false and encode the x and y
#     # dimensions separately
#     raise NotImplementedError

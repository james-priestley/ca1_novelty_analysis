import os
import argparse
import numpy as np
import warnings
import time
import sys
import multiprocessing
import itertools as it
import warnings
import copy
import shutil

from sima import Sequence
from sima import ImagingDataset

from progressbar import ProgressBar
from sima.sequence import _IndexedSequence
from sima.sequence import _MotionCorrectedSequence
#from sima.sequence import _TimeAvgSequence
from sima.motion import PlaneTranslation2D


def _first_obs(seq):
    print('searching for first observed values for each pixel')
    p = ProgressBar(seq.shape[0])
    p.start()

    frame_iter1 = iter(seq)
    first_obs = next(frame_iter1)
    for i, frame in enumerate(frame_iter1):
        p.update(i)
        for frame_chan, fobs_chan in zip(frame, first_obs):
            fobs_chan[np.isnan(fobs_chan)] = frame_chan[np.isnan(fobs_chan)]
        if all(np.all(np.isfinite(chan)) for chan in first_obs):
            break
    p.finish()
    return first_obs


def _fill_gaps(first_obs, frame_iter2):
    """Fill missing rows in the corrected images with data from nearby times.

    Parameters
    ----------
    frame_iter1 : iterator of list of array
        The corrected frames (one list entry per channel).
    frame_iter2 : iterator of list of array
        The corrected frames (one list entry per channel).

    Yields
    ------
    list of array
        The corrected and filled frames.

    """
    most_recent = [x * np.nan for x in first_obs]
    for i, frame in enumerate(frame_iter2):
        for fr_chan, mr_chan in zip(frame, most_recent):
            mr_chan[np.isfinite(fr_chan)] = fr_chan[np.isfinite(fr_chan)]
        yield np.array([np.nan_to_num(mr_ch) + np.isnan(mr_ch) * fo_ch
               for mr_ch, fo_ch in zip(most_recent, first_obs)])


def frame_blocks(frames, chunk_size):
    try:
        t1, t2 = frames
    except TypeError:
        t1 = 0
        t2 = frames

    n_frames = t2-t1
    blocks = np.linspace(
        t1, t2, int(np.ceil(float(n_frames)/chunk_size))+1,
        dtype=int)
    return list(zip(blocks[:-1], blocks[1:]))


def pool_helper(args):
    return args[0](*args[1:])


def frame_writer(out_file, first_obs, dtype, seq_dictionary):
    seq_dictionary = copy.deepcopy(seq_dictionary)
    seq = seq_dictionary.pop('__class__')._from_dict(seq_dictionary)
    num_f = seq._base.shape[0]

    #iterator = _fill_gaps(first_obs, iter(seq))
    iterator = iter(seq)
    shape = (np.prod(seq.shape[1:]), seq.shape[0])
    start = seq._indices[0].start

    f = np.memmap(out_file, mode='r+', dtype=dtype, order='C', # or C? who knows...
                  shape=(shape[0], num_f))

    p = ProgressBar(seq.shape[0])
    p.start()
    Z = 0
    Y = 1
    X = 2 
    C = 3

    for f_idx, frame in enumerate(iterator):
        p.update(f_idx)
        f[:,start+f_idx] = np.clip(
            np.nan_to_num(np.reshape(seq._get_frame(f_idx).transpose((Z, X, Y, C)), 
                shape[0], order='C')), 0, None) # .transpose((C,X,Y,Z))
    p.finish()
    del f
    return out_file


def convert(sequence, out_file, cycle=0, chunk_size=500, n_processes=4, channel='Ch2', dtype=np.single,
            first_obs=150, trim=0, z_plane=None, skip_init=0, n_frames=None, order='C', ):
            # time_avg=False?
    #ds = ImagingDataset.load(in_file)
    #channel = ds._resolve_channel(channel)
    # if time_avg:
    #     sequence = _TimeAvgSequence(ds.sequences[cycle])[:n_frames]
    # else:

        # TODO: multiple sequences
        # TODO: does this handle z correctly?
    #sequence = ds.sequences[0][:n_frames]
    shape = (np.prod(sequence.shape[2:]), sequence.shape[0])
    #seq_dictionary = sequence._todict()
    first_obs = _first_obs(sequence[:first_obs])

    blocks = frame_blocks(shape[1], shape[1]/n_processes)
    sequences = []
    for t1,t2 in blocks:
        sequences.append(sequence[t1:t2]._todict())
#    del ds
    del sequence

    f = np.memmap(out_file, mode='w+', dtype=dtype, order=order, shape=shape)

    # Need this line to avoid acquiring a lock that cannot be released (deadlock) -Z
    with multiprocessing.get_context("spawn").Pool(processes=n_processes) as pool:
        start = time.time()
        num_blocks = len(blocks)
        print('exporting in %i blocks' % (num_blocks))
        pool.map(
            pool_helper,
            zip(it.repeat(frame_writer), it.repeat(out_file), it.repeat(first_obs), it.repeat(dtype),
                sequences))
    print('finished in %i seconds' % (time.time() - start))


def main():
    argParser = argparse.ArgumentParser()
    argParser.add_argument(
        "-s", "--scratch_drive", action="store", type=str,
        default="/scratch/caiman", help="destination drive for memmap file")
    argParser.add_argument(
        "-c", "--channel", action="store", type=str,
        default="Ch2", help="channel name. default is Ch2")
    argParser.add_argument(
        '-n', '--num_frames', action='store', type=int,
        default=None, help='number of frames to convert')
    argParser.add_argument(
        '-t', '--time_avg', action='store_true', default=False,
        help='apply _TimeAvgSqeuence wrapper prior to conversion')
    argParser.add_argument(
        "filename", action="store", type=str, default="",
        help=("sima file to convert to memmap"))
    args = argParser.parse_args()

    ext, filename = os.path.split(args.filename)
    scratch_dir = os.path.join(args.scratch_drive, ext[1:])
    if not os.path.exists(scratch_dir):
        os.makedirs(scratch_dir)

    savefile = os.path.join(
        scratch_dir, f'{os.path.splitext(filename)[0]}.mmap')

    convert(args.filename, savefile,
            channel=args.channel, first_obs=150, n_frames=args.num_frames)

    #sequence = ImagingDataset.load(args.filename).sequences[0]
    ds = ImagingDataset.load(args.filename)
    if args.time_avg:
        sequence = _TimeAvgSequence(ds.sequences[0])[:args.num_frames]
    else:
        sequence = ds.sequences[0][:args.num_frames]

    print(sequence.shape)
    seq = Sequence.create(
        'memmap', savefile,(sequence.shape[2], sequence.shape[3], sequence.shape[0], 1),
        'yxtc')
    sima_file = os.path.join(scratch_dir, filename)
    if os.path.exists(sima_file):
        shutil.rmtree(sima_file)

    print(f'memmap file:\n{savefile}')
    print(f'sima filepath:\n{sima_file}')
    ImagingDataset([seq], sima_file)


if __name__ == '__main__':
    main()

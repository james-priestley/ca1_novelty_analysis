import numpy as np
import itertools as it

from sima.sequence import _Sequence_HDF5


class Struct(object):

    def __init__(self, **entries):
        self.__dict__.update(entries)


def _pool_helper(args):
    return args[0](*args[1:])


def pool_helper(pool, method, *args):
    args = list(args)

    for i, _ in enumerate(args):
        if not hasattr(args[i], '__iter__'):
            args[i] = it.repeat(args[i])

    if pool is None:
        return map(_pool_helper, zip(it.repeat(method), *args))

    return pool.map(_pool_helper, zip(it.repeat(method), *args))


def frame_blocks(frames, chunk_size):
    try:
        t1, t2 = frames
    except TypeError:
        t1 = 0
        t2 = frames

    if chunk_size == 0:
        return [(t1, t2)]

    n_frames = t2-t1
    blocks = np.linspace(
        t1, t2, int(np.ceil(float(n_frames)/chunk_size)+1),
        dtype=int)
    return list(zip(blocks[:-1], blocks[1:]))


def slice_sequence(seq, n_processes=4, max_len=None):
    n_processes = max(1, n_processes-1)

    if max_len is None:
        max_len = np.ceil(seq.shape[0]/n_processes)

    batches = frame_blocks(seq.shape[0],
                           min(max_len, np.ceil(seq.shape[0]/n_processes)))

    return map(
        lambda seq_indices: seq_indices[0][slice(*seq_indices[1])],
        zip(it.repeat(seq), batches))


def bind_locked_get_frame(seq, lock):
    def locked_get_frame(self, t):

        """Get the frame at time t, but not clipped"""
        slices = tuple(slice(None) for _ in range(self._T_DIM)) + (t,)

        with lock:
            for i in range(11):
                try:
                    frame = self._dataset[slices]
                    break
                except Exception:
                    pass

        if i == 10:
            raise Exception

        swapper = [None for _ in range(frame.ndim)]
        for i, v in [(self._Z_DIM, 0), (self._Y_DIM, 1),
                     (self._X_DIM, 2), (self._C_DIM, 3)]:
            if i >= 0:
                j = i if self._T_DIM > i else i - 1
                swapper[j] = v
            else:
                swapper.append(v)
                frame = np.expand_dims(frame, -1)
        assert not any(s is None for s in swapper)
        for i in range(frame.ndim):
            idx = swapper.index(i)
            if idx != i:
                swapper[i], swapper[idx] = swapper[idx], swapper[i]
                frame = frame.swapaxes(i, idx)
        assert swapper == [0, 1, 2, 3]
        assert frame.ndim == 4
        return frame.astype(float)

    h5_seq = seq
    while type(h5_seq) != _Sequence_HDF5:
        h5_seq = h5_seq._base

    bound = locked_get_frame.__get__(h5_seq, h5_seq.__class__)
    setattr(h5_seq, '_get_frame', bound)

    # This is for the _InterpolateGapFilledSequence Sequence wrapper in sima
    # TODO: is there a better way to do this?
    try:
        seq._unmasked_base
    except AttributeError:
        pass
    else:
        bind_locked_get_frame(seq._unmasked_base, lock)

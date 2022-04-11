from sima.sequence import Sequence, _WrapperSequence, _MotionCorrectedSequence
import numpy as np
import inspect
import sima
import sima.sequence
import cv2
import os
import warnings

from scipy.ndimage.filters import gaussian_filter
from numba import float32, int32, int16, njit

from lab3.misc.sima_compatibility import sima_compatible

class _Sequence_Suite2p(Sequence):

    """Sequence for reading binary files in the Suite2p format

    Parameters
    ----------
    ops_path : path to the ops.npy file for a suite2p dataset as a string

    """

    #TODO: multiple channels, multiple frames
    def __init__(self, ops_path):
        self._ops_path = os.path.abspath(ops_path)

        ops = np.load(ops_path, allow_pickle=True)[()]
        self._data_path = os.path.join(ops['save_path'], 'data.bin')
        self._nframes = int(ops['nframes'])
        self._Ly = ops['Ly']
        self._Lx = ops['Lx']
        self._file = open(self._data_path, 'rb')
        self._block_size = self._Ly*self._Lx*2

    def __del__(self):
        self._file.close()

    def __len__(self):
        return self._nframes

    def _get_frame(self, t):
        buff = self._file.read(self._block_size)
        data = np.frombuffer(buff, dtype=np.int16, offset=0)
        data = np.reshape(data, (1, self._Ly, self._Lx), 1)
        return data

    def _todict(self, savedir=None):
        d = {'__class__': self.__class__,
             'ops_path': self._ops_path}

        if savedir is None:
            d.update({'ops_path': os.path.abspath(self._ops_path)})
        else:
            d.update({'_abspath': os.path.abspath(self._ops_path),
                      '_relpath': os.path.relpath(self._ops_path, savedir)})
        return d



class _SpatialFilterSequence(_WrapperSequence):
    """Sequence for gaussian blurring and clipping each frame.

    Parameters
    ----------
    base : Sequence

    """

    def __init__(self, base):
        super(_SpatialFilterSequence, self).__init__(base)

    def _transform(self, frame):
        n_channels = frame.shape[-1]
        ch_frames = []
        for i in xrange(n_channels):
            ch_frame = frame[..., i]
            filtered_frame = gaussian_filter(ch_frame, sigma=3)
            clipped = np.clip(filtered_frame,
                              np.nanpercentile(filtered_frame, 50),
                              np.nanpercentile(filtered_frame, 99.5))

            ch_frames.append(clipped)

        return np.stack(ch_frames, axis=-1)

    def _get_frame(self, t):
        frame = self._base._get_frame(t)
        return np.array(map(self._transform, frame))

    def __iter__(self):
        for frame in self._base:
            yield self._transform(frame)

    @property
    def shape(self):
        return self._base.shape

    def __len__(self):
        return len(self._base)

    def _todict(self, savedir=None):
        return {
            '__class__': self.__class__,
            'base': self._base._todict(savedir),
        }


class _SplicedSequence(_WrapperSequence):
    """Sequence for splicing together nonconsecutive frames
    Parameters
    ----------
    base : Sequence
    times : list of frame indices of the base sequence
    """

    def __init__(self, base, times):
        super(_SplicedSequence, self).__init__(base)
        self._base_len = len(base)
        self._times = times

    def __iter__(self):
        try:
            for t in self._times:
                # Not sure if np.copy is needed here (see _IndexedSequence)
                yield np.copy(self._base._get_frame(t))
        except NotImplementedError:
            if self._indices[0].step < 0:
                raise NotImplementedError(
                    'Iterating backwards not supported by the base class')
            idx = 0
            for t, frame in enumerate(self._base):
                try:
                    whether_yield = t == self._times[idx]
                except IndexError:
                    raise StopIteration
                if whether_yield:
                    # Not sure if np.copy is needed here (see _IndexedSequence)
                    yield np.copy(frame)
                    idx += 1

    def _get_frame(self, t):
        return self._base._get_frame(self._times[t])

    def __len__(self):
        return len(self._times)

    def _todict(self, savedir=None):
        return {
            '__class__': self.__class__,
            'base': self._base._todict(savedir),
            'times': self._times
        }


class _MedianSequence(_WrapperSequence):
    """Sequence for applying a nan median along one of the dimensions.

    Parameters
    ----------
    base : Sequence
    axis : axis to perform nan median along

    """

    def __init__(self, base, axis=0):
        super(_MedianSequence, self).__init__(base)
        self._axis = axis
        self._shape = self._base.shape[:axis + 1] + (1,) + \
                      self._base.shape[axis + 2:]

    def _get_frame(self, t):
        frame = self._base._get_frame(t)
        return np.nanmedian(frame, axis=self._axis, keepdims=True)

    def __iter__(self):
        for frame in self._base:
            yield np.nanmedian(frame, axis=self._axis, keepdims=True)

    @property
    def shape(self):
        return self._shape

    def __len__(self):
        return len(self._base)

    def _todict(self, savedir=None):
        return {
            '__class__': self.__class__,
            'base': self._base._todict(savedir),
            'axis': self._axis}


class _NumPyFuncSequence(_WrapperSequence):
    """Sequence for applying a numypy function to each frame along one of the
    dimensions.

    Parameters
    ----------
    base : Sequence
    func : str containeng the function name to apply i.e. 'max' or 'nanmean'
    axis : axis to perform nan median along

    """

    def __init__(self, base, func, axis=0):
        super(_NumPyFuncSequence, self).__init__(base)
        self._axis = axis
        self._shape = self._base.shape[:axis + 1] + (1,) + \
            self._base.shape[axis + 2:]
        self._func = eval('np.%s' % func)
        self._func_name = func

    def _get_frame(self, t):
        frame = self._base._get_frame(t)
        return self._func(frame, axis=self._axis, keepdims=True)

    def __iter__(self):
        for frame in self._base:
            yield self._func(frame, axis=self._axis, keepdims=True)

    @property
    def shape(self):
        return self._shape

    def __len__(self):
        return len(self._base)

    def _todict(self, savedir=None):
        return {
            '__class__': self.__class__,
            'base': self._base._todict(savedir),
            'func': self._func_name,
            'axis': self._axis}


class _ClippedSequence(_WrapperSequence):
    """Sequence for clipping frame fluorescence to min and max values.

    Parameters
    ----------
    base : Sequence
    min_val : Minimum fluorescence value
    max_val: Maximum fluorescence value

    """

    def __init__(self, base, min_val, max_val):
        super(_ClippedSequence, self).__init__(base)
        # self._clip_val=99.5
        self._min_val = min_val
        self._max_val = max_val

    def _get_frame(self, t):
        frame = self._base._get_frame(t)
        return np.clip(frame, self._min_val, self._max_val)

    def __iter__(self):
        for frame in self._base:
            yield np.clip(frame, self._min_val, self._max_val)

    def __len__(self):
        return len(self._base)

    def _todict(self, savedir=None):
        return {
            '__class__': self.__class__,
            'base': self._base._todict(savedir),
            'min_val': self._min_val,
            'max_val': self._max_val
        }


class _Suite2pRigidSequence(_MotionCorrectedSequence):
    """

    Sequence wrapper for the rigid transformation from Suite2P

    Parameters
    --------------
    base : Sequence
        base sequence
    ops : dict or list[dict]
        Suite2p options dictionaries which stores the displacements

    """

    def __init__(self, base, ops=None, displacements=None):

        if ops is not None:
            displacements = self._process_ops(ops)
        super(_Suite2pRigidSequence, self).__init__(base, displacements)

    def _process_ops(self, ops):
        if len(ops) > 1:
            all_planes = []
            for plane in ops:
                all_planes.append(self._recover_displacements(plane))
            displacement = np.stack(all_planes, axis=1)
            print(displacement.shape)
            return displacement
        return self._recover_displacements(ops[0])

    def _recover_displacements(self, ops):
        _displacements = np.stack(
            [ops['yoff'], ops['xoff']]) * -1  # get the data of the x and y displacement and apply a flip
        # use the minimum displacement to translate and clip to absolute value to make sure that there are no
        # negative values, and transpose
        _displacements = np.stack([np.round(d + np.abs(np.min(d))).astype('int') for d in _displacements]).T
        # restructure the displacement shape to be according to requirements
        _displacements = _displacements.reshape(-1, 1, 2)

        return _displacements

    def _todict(self, savedir=None):
        return {
            '__class__': self.__class__,
            'base': self._base._todict(savedir),
            'displacements': self.displacements.astype('int16')
        }

#@sima_compatible
class _NoRMCorreRigidSequence(_MotionCorrectedSequence):
    """

    Sequence wrapper for the rigid transformation from NoRMCorre

    Parameters
    --------------
    base : Sequence
        base sequence
    mc : NoRMCorre motion correction object
        NoRMCorre object which stores the displacements

    """

    def __init__(self, base, mc=None, displacements=None):
        # from pudb import set_trace 
        # set_trace()
        if mc is not None:
            displacements = self._recover_displacements(mc, 
                num_planes=base.shape[1])
        super().__init__(base, displacements)

    @classmethod
    def _recover_displacements(cls, mc, num_planes=1):

        _displacements = np.array(mc.shifts_rig)

        if num_planes > 1:
            max_dz = np.max(np.abs(_displacements[:,2]))
            if max_dz > 0:
                warnings.warn(f"z-motion of up to {max_dz}px detected! "
                                f"SIMA does not handle z motion!")

            _displacements = np.repeat(_displacements[:, np.newaxis, :2], axis=1, 
                                        repeats=num_planes)
        else:
            _displacements = _displacements[:, np.newaxis, :]

        _displacements = np.round(_displacements - np.min(_displacements)).astype('int16')

        return _displacements

    def _todict(self, savedir=None):
        return {
            '__class__': self.__class__,
            'base': self._base._todict(savedir),
            'displacements': self.displacements.astype('int16')
        }


# TODO : this does not actually work
# class _BlockInterpSequence(_WrapperSequence):
#     """For storing Suite2p non-rigid motion correction results using SIMA.

#     Suite2p non-rigid correction partitions the data into XY blocks that are
#     independently motion corrected via plane translation. Here those blockwise
#     shifts are bilinearly interpolated to calculate a globally nonlinear
#     correction, applied as a piecewise affine transform.

#     TODO : Multiplane corrections. Currently we expect only one set of
#     displacements and apply these to all frames.

#     Parameters
#     ----------
#     base : Sequence
#     ops : dict
#         Suite2p options dictionary, which stores the block displacements

#     """

#     def __init__(self, base, ops):
#         super(_BlockInterpSequence, self).__init__(base)
#         self._ops = ops

#         # get block information
#         nblocks = ops['nblocks']
#         yblock = ops['yblock'][::-1]
#         xblock = ops['xblock'][::-1]

#         # reshape and store displacements
#         yshifts = (ops['yoff1'] * -1).astype('float32').reshape(
#             -1, nblocks[0], nblocks[1])
#         xshifts = (ops['xoff1'] * -1).astype('float32').reshape(
#             -1, nblocks[0], nblocks[1])
#         self.displacements = np.stack([yshifts, xshifts], axis=-1)

#         num_rows, num_columns = self._base.shape[2:4]

#         # save mesh grids for later interpolations
#         yb = np.array(yblock[::nblocks[1]]).mean(axis=1).astype(np.float32)
#         iy = np.arange(0, num_rows, 1, np.float32)
#         iy = np.interp(iy, yb, np.arange(0, yb.size, 1, int)).astype(np.float32)

#         xb = np.array(xblock[:nblocks[1]]).mean(axis=1).astype(np.float32)
#         ix = np.arange(0, num_columns, 1, np.float32)
#         ix = np.interp(ix, xb, np.arange(0, xb.size, 1, int)).astype(np.float32)

#         self.imshx, self.imshy = np.meshgrid(ix, iy)
#         self.mshx, self.mshy = np.meshgrid(
#             np.arange(0, num_columns, 1, np.float32),
#             np.arange(0, num_rows, 1, np.float32))

#     def __len__(self):
#         return len(self._base)

#     def _transform(self, frame, t):
#         # upsample block shifts
#         yup = bilinear_transform(
#             self.displacements[t][..., 0], self.imshy.copy(),
#             self.imshx.copy()).astype('float32')
#         xup = bilinear_transform(
#             self.displacements[t][..., 1], self.imshy.copy(),
#             self.imshx.copy()).astype('float32')

#         out = np.nan * np.ones(frame.shape)
#         for pidx, plane in enumerate(frame):
#             for cidx in range(plane.shape[-1]):
#                 out[pidx, :, :, cidx] = bilinear_transform(
#                     plane[..., cidx].astype('int16'),
#                     self.mshy + yup, self.mshx + xup)

#         return out
    
#     def __iter__(self):
#         for frame in self._base:
#             yield self._transform(frame)

#     def _get_frame(self, t):
#         return self._transform(self._base._get_frame(t), t)  # add args
    
#     def _todict(self, savedir=None):
#         return {
#             '__class__': self.__class__,
#             'base': self._base._todict(savedir),
#             'ops': self._ops
#         }


class _InterpolateGapFilledSequence(_WrapperSequence):
    """Sequence for doing random stuff.

    Parameters
    ----------
    base : Sequence
    kernel_size : Size of 2D gaussian kernel to use for interpolating nans.

    """

    def __init__(self, base, kernel_size):
        super(_InterpolateGapFilledSequence, self).__init__(base)

        from astropy.convolution import Gaussian2DKernel
        from astropy.convolution import interpolate_replace_nans

        self._kernel_size = kernel_size
        self._shape = self._base.shape
        self._interp_fn = interpolate_replace_nans
        self._kernel = Gaussian2DKernel(self._kernel_size)
        self._cutoff = np.prod(self._shape[1:])/3

        seq_dict = base._todict()
        self._unmasked_base = seq_dict.pop('__class__')._from_dict(seq_dict)
        seq_copy = self._unmasked_base
        self._masked = True
        while True:
            if isinstance(seq_copy._base, sima.sequence._MaskedSequence):
                    break

            try:
                seq_copy._base._base
                seq_copy = seq_copy._base
            except:
                self._masked = False
                break

        if self._masked:
            seq_copy._base = seq_copy._base._base
            self.__iter__ = self._iter_masked
            self._get_frame = self._get_frame_masked
        else:
            self._unmasked_base = None

    def _transform_masked(self, frame_masked, frame_unmasked):
        nan_locations = np.where(np.isnan(frame_masked))[0].shape[0]
        if nan_locations > self._cutoff:
            return frame_masked

        frame_mask = np.where(
            np.logical_and(np.isnan(frame_masked), np.isfinite(frame_unmasked)))

        _frame = np.empty(frame_unmasked.shape)
        for p, plane in enumerate(frame_unmasked):
            for c, channel in enumerate(np.rollaxis(plane, 2, 0)):
                _frame[p, :, :, c] = self._interp_fn(channel, self._kernel)
        _frame[frame_mask] = np.nan
        return _frame

    def _get_frame_masked(self, t):
        frame_masked = self._base._get_frame(t)
        frame_unmasked = self._unmasked_base._get_frame(t)
        return self._transform_masked(frame_masked, frame_unmasked)

    def _iter_masked(self):
        for frame_masked, frame_unmasked in zip(self._base,
                                                self._unmasked_base):
            yield self._transform_masked(frame_masked, frame_unmasked)

    def _transform(self, frame):
        _frame = np.empty(frame.shape)
        for p, plane in enumerate(frame):
            for c, channel in enumerate(np.rollaxis(plane, 2, 0)):
                _frame[p, :, :, c] = self._interp_fn(channel, self._kernel)
        return _frame

    def _get_frame(self, t):
        frame = self._base._get_frame(t)
        return self._transform(frame)

    @property
    def shape(self):
        return self._shape

    def __len__(self):
        return len(self._base)

    def _todict(self, savedir=None):
        return {
            '__class__': self.__class__,
            'base': self._base._todict(savedir),
            'kernel_size': self._kernel_size
        }


@njit(['(int16[:, :],float32[:,:], float32[:,:])',
       '(float32[:, :],float32[:,:], float32[:,:])'])
def bilinear_transform(I, yc, xc):
    """Doc string"""

    I_transform = np.zeros((yc.shape))

    num_rows, num_columns = I.shape
    yc_floor, xc_floor = [c.copy().astype(np.int32) for c in [yc, xc]]
    yc -= yc_floor
    xc -= xc_floor

    # calculate value of each pixel
    for i in range(yc_floor.shape[0]):
        for j in range(yc_floor.shape[1]):
            yf = min(num_rows - 1, max(0, yc_floor[i, j]))
            xf = min(num_columns - 1, max(0, xc_floor[i, j]))
            yf1 = min(num_rows - 1, yf + 1)
            xf1 = min(num_columns - 1, xf + 1)
            I_transform[i, j] = \
                np.float32(I[yf, xf]) * (1 - yc[i, j]) * (1 - xc[i, j]) \
                + np.float32(I[yf, xf1]) * (1 - yc[i, j]) * xc[i, j] \
                + np.float32(I[yf1, xf]) * yc[i, j] * (1 - xc[i, j]) \
                + np.float32(I[yf1, xf1]) * yc[i, j] * xc[i, j]

    return I_transform


class _AffineRegisteredSequence(_WrapperSequence):
    """Wraps a SIMA sequence to apply affine warp matrix. Best on 
    already MC-wrapped sequences that are aligned to themselves such 
    that a single warp matrix is appropriate for the whole sequence.

    Parameters
    __________
    base : SIMA sequence, typically already MC'd
    warpMat : warp matrix to transform sequence into alignment
        with the template image used for cross-registration. Warp
        matrices for all wrapped sequences within an affine cross-
        registered dataset will align their respective sequences to
        the same shape (to whatever extent possible).
    ECC : enhanced correlation coefficient (calculated by 
        cv2.findTransformECC) which reflects goodness of fit between
        transformed image and the reference it was mapped to. value of
        -1 indicates the ML algorithm (maximizing ECC as obj func) 
        failed to converge.
    """

    def __init__(self, base, warpMat, ECC, refDims):
        super(_AffineRegisteredSequence, self).__init__(base)

        self.warpMat = warpMat
        self.ECC = float(ECC)
        self.refDims = refDims

        if ECC == -1:
            raise ValueError('warpMat == identity matrix. No transform will occur.')


    @property
    def _frame_shape(self):
        return self._frame_shape #inherited from MC wrapper

    def __len__(self):
        return len(self._base)


    def __iter__(self):
        for frame in self._base:
            yield self._align(frame)


    #return 4D aligned frame (zyxc)
    def _align(self, frame):
        assert len(frame.shape) == 4 #zyxc
        nPlanes = frame.shape[0]
        nChannels = frame.shape[3]
        alignedFrame = np.zeros((
            nPlanes, self.refDims[0], self.refDims[1], nChannels))
        for p in np.arange(nPlanes):
            for c in np.arange(nChannels):
                currXY = frame[p, :, :, c]
                alignedXY = cv2.warpAffine(currXY, self.warpMat, 
                    (self.refDims[1], self.refDims[0]), 
                    flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
                alignedFrame[p, :, :, c] = alignedXY

        return alignedFrame

    def _get_frame(self, t):
        return self._align(self._base._get_frame(t))

    def _todict(self, savedir=None):
        return{
        '__class__': self.__class__,
        'base': self._base._todict(savedir),
        'warpMat': self.warpMat,
        'ECC': self.ECC,
        'refDims': self.refDims
        }


sima.sequence.__dict__.update(
    {k: v for k, v in locals().items() if
        inspect.isclass(v) and (issubclass(v, sima.sequence._WrapperSequence) or
        issubclass(v, sima.sequence.Sequence))})

import os
import numpy as np
import scipy.sparse as sparse
import h5py
import warnings

from packaging import version

# THIS PACKAGE IS DEPRECATED! 
# My environment does not support h5py >= 2.9 so I use this to control
# caching, but use h5py native cache if possible. -Z
H5_VERSION = h5py.__version__
if version.parse(H5_VERSION) < version.parse("2.9"):
    warnings.warn(f"h5py version {H5_VERSION} detected, "
                f"using deprecated `h5py_cache` package! "
                "(upgrade to h5py >= 2.9 to bypass this warning)")
    import h5py_cache

import sima
from sima import ImagingDataset
from sima.sequence import _Sequence_HDF5

from ...experiment.base import ImagingOnlyExperiment
from ...misc.sima_helpers import backup_dir
from . import trefide_denoiser 
from . import convertToMemmap

FLOATSIZE = 8 # bytes
H5_CHUNKSIZE = 10*10**6 # bytes
N_CHUNKS = 10

FASTSCRATCH = "/fastscratch/memmaps"

def import_denoised_signals(expt):
    denoised_ds = DenoisedDataset.load(expt.sima_path)
    denoised_expt = ImagingOnlyExperiment(denoised_ds.denoised_savedir)

    with denoised_expt.signals_file(mode='r') as origin, \
        expt.signals_file(mode='a') as target:

        for k in origin.keys():
            print(f'Importing {k}...')
            target.put(k, origin[k])

    print('Importing ROIs...')
    expt.imaging_dataset.add_ROIs(denoised_expt.roi_list(label='denoised'), 'denoised')

class _DenoisedSequence(_Sequence_HDF5):
    """This works in a FUNDAMENTALLY DIFFERENT way from other sima 
    sequences. Rather than storing every frame, this stores the spatial 
    and temporal information of the session separately (similar to an SVD)
    and reconstitutes them on the fly as needed.

    This allows compression to occur, as biologically-relevant signals
    occupy a much lower dimensional manifold in pixel space than 2**18 (=512 x 512)

    If Y_yxt is a tensor representing the movie, then we have
        Y_yxt = U_yxk.V^k_t
    where k is the latent dimension, with log10(k) ~ 2. 
    U and V are stored instead of Y.

    (Don't put @sima_compatible here, it's not going to work.)
    """
    def __init__(self, path, group='/denoised0'):
        self._path = os.path.abspath(path)
        if version.parse(H5_VERSION) >= version.parse("2.9"):
            # Use h5py-native caching (>= 2.9 only)
            self._file = h5py.File(path, 'r', 
                                    rdcc_nbytes=N_CHUNKS*H5_CHUNKSIZE, # cache optimizations for performance
                                    rdcc_nslots=100*N_CHUNKS)
        else:
            # Use deprecated h5py_cache package
            self._file = h5py_cache.File(path, 'r', 
                            chunk_cache_mem_size=N_CHUNKS*H5_CHUNKSIZE) #  # h5py

        self._group = self._file[group]
        # self._spatial_shape = self._group['U'].shape[:-1]
        # self._nk = self._group['U'].shape[-1]

        # This is needed for every frame so just load it into RAM
        U = np.array(self._group['U'])
        self._spatial_shape = U.shape[:-1]
        self._nk = U.shape[-1]
        resh = U.reshape(np.product(self._spatial_shape), self._nk)
        self._spatial = sparse.csr_matrix(resh)
        # self._spatial = sparse.csr_matrix(np.array(self._group['U']).reshape( 
        #                     (np.product(self._spatial_shape), self._nk))) # optimizations... 
        self._temporal = np.array(self._group['V'])
        #self._temporal = np.array(self._group['V'])
        
    def __len__(self):
        return self._temporal.shape[1]
    
    def _get_frame(self, t):
        """This is where the reconstitution happens.
        Shape of U is yxk, shape of V is kt
        """
        flat = self._spatial @ self._temporal[:, t]
        return flat.reshape((1,) + self._spatial_shape + (1,))
    
    def _todict(self, savedir=None):
        d = {'__class__': self.__class__,
             'group': self._group.name}
        if savedir is None:
            d.update({'path': os.path.abspath(self._path)})
        else:
            d.update({'_abspath': os.path.abspath(self._path),
                      '_relpath': os.path.relpath(self._path, savedir)})
        return d

class DenoisedDataset(ImagingDataset):
    """Interface for using Trefide denoiser with SIMA. DenoisedDatasets
    are instantiated via the load method, inherited from ImagingDataset. They
    are not initialized directly. You should only initialize a DenoisedDataset
    from an already motion-corrected dataset.

    Examples
    --------
    >>> from lab3.extraction.denoising import DenoisedDataset
    >>> ds = DenoisedDataset.load("/path/to/my/folder.sima")
    >>> ds.denoise()

    Denoised datasets are saved to a *NEW* h5 and sima folder 
    (/old/path/OLDNAME_denoised.{h5|sima}) as the underlying data 
    is changed.

    This package also implements compression.

    See Buchanan et al. 2019 for details. 
    https://www.biorxiv.org/content/10.1101/334706v4
    """

    def denoise(self, overwrite=False, reprocess=True, on_scratch=True, first_obs=1, 
                signal_channel='0', block_height=40, block_width=40, t_sub=2,  
                n_processes=8, sima_compatible=False, strict_checking=False, **kwargs):
        """Run Trefide denoising on the imaging dataset. It should ordinarily not be 
        necessary to change the Trefide parameters from defaults.

        Parameters
        ----------
        sima_compatible : bool, optional
            Whether to store the denoised HDF5 in full format (i.e., all frames) 
            or in U, V format. The advantage of sima_compatible is you can watch the 
            denoised video on the vis_server. Default False.
        strict_checking : bool, optional
            Whether to validate the data. Default False.
        overlapping : bool, optional
            Whether to overlap blocks the video is split into. Default True.
        block_height : int, optional 
            Frame height will be trimmed to a multiple of this number. Default 40.
        block_width: int, optional
            Frame width will be trimmed to a multiple of this number. Default 40.
        d_sub : int, optional
            Factor by which to downsample spatial dimensions. Default 2. 
        t_sub : int, optional 
            Factor by which to downsample time. Default 2. 
        max_components : int, optional
            Cap on the maximum number of components. Default 50. 
        max_iters_main : int, optional
            Cap on maximum number of iterations in main method. Default 10. 
        max_iters_init : int, optional
            Cap on maximum number of initialization iterations. Default 40. 
        consec_failures : int, optional
            Number of consecutive failures to tolerate. Default 3. 
        tol : float, optional
            Tolerance. Default 5e-3. 
        enable_temporal_denoiser : bool, optional
            Whether to use temporal denoising. Default True. 
        enable_spatial_denoiser : bool, optional
            Whether to use spatial denoising. Default True.

        See also
        --------
        lab3.extraction.denoising.trefide_denoiser.denoise
        """
        
        sequences = []
        
        for i, seq in enumerate(self.sequences):
            try:
                self.validate(seq)
            except AssertionError:
                if strict_checking:
                    raise
                else:
                    pass

            self.mmapname = self.write_memmap(seq, block_height=block_height, 
                                            block_width=block_width, t_sub=t_sub, 
                                            overwrite=overwrite, reprocess=reprocess, 
                                            on_scratch=on_scratch, first_obs=first_obs, 
                                            signal_channel=signal_channel, n_processes=n_processes)

            new_seq = self.denoise_mmap(self.mmapname, i, sima_compatible=sima_compatible,
                                        block_height=block_height, block_width=block_width,
                                        t_sub=t_sub, **kwargs)
            
            sequences.append(new_seq)

        if os.path.exists(self.denoised_savedir):
            backup_dir(self.denoised_savedir, delete_original=True)
            
        if sima_compatible:            
            new_ds = ImagingDataset(sequences, self.denoised_savedir, 
                                channel_names=self.channel_names)
        else:
            new_ds = self.__class__(sequences, self.denoised_savedir, 
                    channel_names=self.channel_names)

        self.cleanup()

    def denoise_mmap(self, mmapname, i, block_height, block_width, t_sub, 
                    sima_compatible=False, **kwargs):
        U, V, mov_denoised = trefide_denoiser.denoise(self.mmapname, block_height=block_height, 
                                                    block_width=block_width, t_sub=t_sub,
                                                    sima_compatible=sima_compatible, **kwargs)
        
        if os.path.exists(self.denoised_h5):
            backup_dir(self.denoised_h5, delete_original=True)

        self.save_denoised_h5(i, U, V, mov_denoised=mov_denoised)
        
        if sima_compatible:
            new_seq = sima.Sequence.create('HDF5', self.denoised_h5, 'yxt', key=f'imaging{i}')
        else:
            new_seq = _DenoisedSequence(self.denoised_h5, group=f"/denoised{i}")

        return new_seq
            
    @property
    def denoised_savedir(self):
        savedir = self.savedir
        if 'denoised' not in savedir:
            # IMPORTANT! *DON'T* write this to a normal sima
            savedir = os.path.splitext(os.path.normpath(savedir))[0] + '_denoised.sima'
        
        return savedir
    
    @property
    def denoised_h5(self):
        
        root, base = os.path.split(self.denoised_savedir)
        base = os.path.splitext(base)[0]
        return os.path.join(root, f'{base}.h5')
        
    def save(self, savedir=None):
        super().save(savedir=self.denoised_savedir)
        
    def save_denoised_h5(self, i, U, V, mov_denoised=None):
        
        with h5py.File(self.denoised_h5, 'a') as f:
            try:
                grp = f[f'denoised{i}']
            except KeyError:
                grp = f.create_group(f"denoised{i}")
            
            grp['U'] = np.pad(U, ((self.bottom, self.top), (self.left, self.right), (0,0)), 
                                constant_values=np.nan)

            V = np.pad(V, ((0,0), (self.beginning, 0)), constant_values=np.nan)

            kdim, tdim = V.shape
            t_chunk_size = int(H5_CHUNKSIZE // (kdim*FLOATSIZE))
            t_chunk_size = min(t_chunk_size, tdim)

            Vdset = grp.create_dataset("V", data=V, 
                                        chunks=(kdim, t_chunk_size))
            
            if mov_denoised is not None:
                try:
                    grp = f[f'/']                
                except KeyError:
                    grp = f.create_group(f"/")
                
                mov_dset = grp.create_dataset(f"imaging{i}", 
                                            shape=U.shape[:-1] + V.shape[1:])

                # TODO: @jit doesn't work with h5py unfortunately
                # EITHER store as memmap (wasteful of hard disk space) or
                # execute as python (slow) or write custom parallel I/O (fiddly)
                # or store U and V and create memmap when needed
                trefide_denoiser.rebuild_movie(U, V, out=mov_dset)
                #grp[f"imaging{i}"] = mov_denoised

    def validate(self, seq):
        data_size = np.product(seq.shape) * FLOATSIZE
        print(f"The whole sequence ({data_size//10**6}MB) will be loaded into RAM")

        assert seq.shape[1] == seq.shape[-1] == 1, \
            "Denoising currently only supports single-plane, single-channel datasets " \
            "(but see the Zequence wrapper if you want to do this)"

        assert seq.shape[0] > seq.shape[1], \
            "Sequence must have at least as many frames as pixels, otherwise this will cause SegFault! "\
            "Do not ignore this warning!"
        
        assert isinstance(seq, sima.sequence._MotionCorrectedSequence), \
            "Is this data motion corrected? Denoising will only give meaningful results " \
            "for motion corrected sequences"

    def write_memmap(self, seq, block_height, block_width, t_sub, 
                    overwrite=False, reprocess=True, on_scratch=True, 
                    first_obs=50, signal_channel='Ch2', n_processes=8):

        test_frame = seq._get_frame(0)
        _, y_valid, _ = np.where(~np.all(np.isnan(test_frame), axis=1))
        ymin, ymax = y_valid[0], y_valid[-1]
        print(ymin, ymax)

        _, x_valid, _ = np.where(~np.all(np.isnan(test_frame), axis=2))
        xmin, xmax = x_valid[0], x_valid[-1]
        print(xmin, xmax)

        seq = seq[:, :, ymin:ymax, xmin:xmax, :]        

        fov_height = (seq.shape[2]//block_height)*block_height
        fov_width = (seq.shape[3]//block_width)*block_width
        fov_depth = 1
        num_frames = (seq.shape[0] // t_sub)*t_sub
        name = os.path.basename(os.path.splitext(os.path.normpath(self.savedir))[0])
        name = name.replace('_', '-') # '_' is significant to CaImAn!

        mmapname = f"{name}_d1_{fov_height}_d2_{fov_width}_d3_{fov_depth}_order_F_frames_{num_frames}_.mmap"

        if on_scratch:
            mmapname = os.path.join(FASTSCRATCH, mmapname)

        # TODO: convert assumes sequence 0
        if not os.path.exists(mmapname) or overwrite:
            btrim = (seq.shape[2] - fov_height)//2    
            ltrim = (seq.shape[3] - fov_width)//2
            ttrim = seq.shape[0] - num_frames

            self.bottom = ymin+btrim
            self.top = test_frame.shape[1] - (fov_height+self.bottom)
            self.left = xmin+ltrim 
            self.right = test_frame.shape[2] - (fov_width+self.left)
            self.beginning = ttrim

            #real_mmap = mmap.transpose((3,2,0,1))[btrim:btrim+fov_height, ltrim:ltrim+fov_width, ttrim:, 0]

            convertToMemmap.convert(seq[ttrim:, :, btrim:btrim+fov_height, ltrim:ltrim+fov_width],
                                    mmapname, n_processes=n_processes, channel=signal_channel, dtype=np.double,
                                    first_obs=first_obs, n_frames=self.num_frames, order='F')
        elif reprocess:
            pass # use existing mmap
        else:
            raise OSError(f"`{self.mmapname}` already exists (reprocess=True to extract from "\
                            f"existing .mmap or overwrite=True to overwrite)")

        return mmapname

    def cleanup(self):
        os.remove(self.mmapname)
        
    def __repr__(self):
        return ('<DenoisedDataset: '
                + f'num_sequences={self.num_sequences},'
                + f'frame_shape={self.frame_shape}, '
                + f'num_frames={self.num_frames}>')


    @property
    def mmapname(self):
        try:
            return self._mmapname
        except AttributeError:
            name = os.path.basename(os.path.splitext(os.path.normpath(self.savedir))[0])
            name = name.replace('_', '-') # '_' is significant to CaImAn!

            z, y, x, c = self.frame_shape
            t = self.num_frames

            mmapname = f"{name}_d1_{y}_d2_{x}_d3_{z}_order_F_frames_{t}_.mmap"
            self._mmapname = mmapname
            return mmapname

    @mmapname.setter 
    def mmapname(self, value):
        self._mmapname = value

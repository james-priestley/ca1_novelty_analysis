# General Dependencies
import os
import numpy as np
import h5py
import warnings

from numba import jit

import sima
from sima import ImagingDataset
from sima.sequence import _Sequence_HDF5

warnings.warn(
"""
Did you link mkl libraries with e.g.
    export LD_PRELOAD=/opt/intel/mkl/lib/intel64/libmkl_def.so:/opt/intel/mkl/lib/intel64/libmkl_avx2.so:/opt/intel/mkl/lib/intel64/libmkl_core.so:/opt/intel/mkl/lib/intel64/libmkl_intel_lp64.so:/opt/intel/mkl/lib/intel64/libmkl_intel_thread.so:/opt/intel/lib/intel64_lin/libiomp5.so
"""
)

"""
CHECK IF RUNNING IN NOTEBOOK TO PRE-EMPT ERROR
"""
try:
    ip = get_ipython()
    if ip.has_trait('kernel'):
        warnings.warn(
"""
###############################################################################
You seem to be running Trefide in a Jupyter notebook. Trefide doesn't play nice 
with Jupyter for reasons I don't understand. Known issues include:
    - Importing fails unless the compiled shared objects (.so) are in the 
    same directory as the .ipynb (doesn't happen with plain command line)
    - Parallelization doesn't work
    - Sometimes you get SegFaults
Proceed with caution... -Z
###############################################################################
"""
            )
except:
    pass

# Preprocessing Dependencies
from trefide.utils import psd_noise_estimate

# PMD Model Dependencies
from trefide.pmd import batch_decompose,\
                        batch_recompose,\
                        overlapping_batch_decompose,\
                        overlapping_batch_recompose,\
                        determine_thresholds
from trefide.reformat import overlapping_component_reformat, weighted_component_reformat

def parse_mmapname(name):
    name = os.path.basename(name)

    parts = name.split('_')
    assert parts[-1] == '.mmap'
    data = {
        'T': int(parts[-2]),
        'order': parts[-4],
        'z': int(parts[-6]),
        'x': int(parts[-8]),
        'y': int(parts[-10]),
        'name': '_'.join(parts[:-10])
    }

    data['shape'] = (data['T'], data['z'], data['x'], data['y'])

    return data 

def trim_sequence_to_block(seq, block_height, block_width, tsub):
    """Sequence size must evenly divide block_height and width
    """
    fov_height = (seq.shape[2]//block_height)*block_height
    fov_width = (seq.shape[3]//block_width)*block_width
    num_frames = (seq.shape[0] // tsub)*tsub


    btrim = (seq.shape[2] - fov_height)//2    
    ltrim = (seq.shape[3] - fov_width)//2
    ttrim = seq.shape[0] - num_frames
    
    mov = np.array(seq).transpose((2,3,0,1,4))[btrim:btrim+fov_height, ltrim:ltrim+fov_width, ttrim:, 0, 0]
    mov = np.ascontiguousarray(np.nan_to_num(mov))

    return mov

def load_mmap_trimmed(mmapname, block_height, block_width, t_sub):

    mmap_info = parse_mmapname(mmapname)
    mmap = np.memmap(mmapname, shape=mmap_info['shape'], 
                        order='F', mode='r') # F BECAUSE TRANSPOSE!

    fov_height = (mmap.shape[2]//block_height)*block_height
    fov_width = (mmap.shape[3]//block_width)*block_width
    num_frames = (mmap.shape[0] // t_sub)*t_sub


    btrim = (mmap.shape[2] - fov_height)//2    
    ltrim = (mmap.shape[3] - fov_width)//2
    ttrim = mmap.shape[0] - num_frames

    real_mmap = mmap.transpose((3,2,0,1))[btrim:btrim+fov_height, ltrim:ltrim+fov_width, ttrim:, 0]
    print(real_mmap.flags)
    return real_mmap

def load_mmap_raw(mmapname):
    mmap_info = parse_mmapname(mmapname)
    mmap = np.memmap(mmapname, shape=mmap_info['shape'], dtype=np.double,
                        order='F', mode='r+')
    return mmap.transpose((3,2,0,1))[...,0]

def denoise(mmapname, sima_compatible=False, max_components=50, max_iters_main=10, max_iters_init=40, 
                d_sub=2, t_sub=2, consec_failures=3, tol=5e-3, block_height=40, 
                block_width=40, overlapping=True, enable_temporal_denoiser=True, 
                enable_spatial_denoiser=True):

    """Run Trefide denoising on the imaging dataset. It should ordinarily not be 
    necessary to change the Trefide parameters from defaults.

    Parameters
    ----------
    mmapname : str
        Path to memmap containing data to denoise.
    sima_compatible : bool, optional
        Whether to reconstitute the full format (i.e., all frames). Default False.
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
    """

    #mov = load_mmap_trimmed(mmapname, block_height, block_width, t_sub)
    mov = load_mmap_raw(mmapname) #TESTING ONLY
    # mov = trim_sequence_to_block(seq, block_height, block_width)
    fov_height, fov_width, num_frames = mov.shape 

    # Calculate denoising thresholds
    print("Calculating thresholds...")
    spatial_thresh, temporal_thresh = determine_thresholds((fov_height, fov_width, min(num_frames, 3000)), # don't spend forever doing this
                                                   (block_height, block_width),
                                                   consec_failures, max_iters_main, 
                                                   max_iters_init, tol, 
                                                   d_sub, t_sub, 5, False,
                                                   enable_temporal_denoiser,
                                                   enable_spatial_denoiser)

    print(spatial_thresh, temporal_thresh)

    if not overlapping:    # Blockwise Parallel, Single Tiling
        print("Decomposing blockwise in parallel using single tiling...")
        spatial_components,\
        temporal_components,\
        block_ranks,\
        block_indices = batch_decompose(fov_height, fov_width, num_frames,
                                        mov, block_height, block_width,
                                        spatial_thresh, temporal_thresh,
                                        max_components, consec_failures,
                                        max_iters_main, max_iters_init, tol,
                                        d_sub, t_sub,
                                        enable_temporal_denoiser, enable_spatial_denoiser)

        U, V = weighted_component_reformat(spatial_components, temporal_components, 
                                            block_ranks, block_indices, 1)
    else:    # Blockwise Parallel, 4x Overlapping Tiling
        print("Decomposing blockwise in parallel using 4x overlapping tiling...")
        spatial_components,\
        temporal_components,\
        block_ranks,\
        block_indices,\
        block_weights = overlapping_batch_decompose(fov_height, fov_width, num_frames,
                                                    mov, block_height, block_width,
                                                    spatial_thresh, temporal_thresh,
                                                    max_components, consec_failures,
                                                    max_iters_main, max_iters_init, tol,
                                                    d_sub, t_sub,
                                                    enable_temporal_denoiser, enable_spatial_denoiser)
        
        print("Reformatting to compressed format...")
        U, V = overlapping_component_reformat(fov_height, fov_width, num_frames,
                                  block_height, block_width,
                                  spatial_components,
                                  temporal_components,
                                  block_ranks,
                                  block_indices,
                                  block_weights)
    
    # if sima_compatible:
    #     print("Rebuilding movie...")


    #     # if not overlapping:  # Single Tiling (No need for reweighting)
    #     #     mov_denoised = np.asarray(batch_recompose(spatial_components,
    #     #                               temporal_components,
    #     #                               block_ranks,
    #     #                               block_indices))
    #     # else:   # Overlapping Tilings With Reweighting
    #     #     mov_denoised = np.asarray(overlapping_batch_recompose(fov_height, fov_width, num_frames,
    #     #                               block_height, block_width,
    #     #                               spatial_components,
    #     #                               temporal_components,
    #     #                               block_ranks,
    #     #                               block_indices,
    #     #                               block_weights)) 
    # else:
    mov_denoised = None


    return U, V, mov_denoised


@jit
def rebuild_movie(U, V, out=None):
    if out is None:
        return U.dot(V)
    else:
        # This is optimized for OOC operations!
        assert out.shape == U.shape[:-1] + V.shape[-1:]
        assert U.shape[-1] == V.shape[0]
        assert not np.any(out)
        k, T = V.shape
        for i in range(k):
#            for t in range(T):
            out += U[...,i:i+1].dot(V[i:i+1]) # ,t

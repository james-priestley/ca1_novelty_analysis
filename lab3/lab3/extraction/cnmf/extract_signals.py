import os
import cv2
import glob
import psutil
import numpy as np
import scipy
import sys
import time

from scipy.ndimage.filters import gaussian_filter
from skimage.external.tifffile import TiffFile

try:
    cv2.setNumThreads(0)
except:
    pass

import caiman as cm
from caiman.motion_correction import MotionCorrect, tile_and_correct, motion_correction_piecewise

import caiman.source_extraction.cnmf as cnmf
from caiman.components_evaluation import evaluate_components, estimate_components_quality_auto

def motion_correct(memmap_name, dview, is3D=False,
                  max_shifts=(6, 6), strides=(48, 48), overlaps=(24, 24), 
                  num_frames_split=100,
                  max_deviation_rigid=3, 
                  pw_second_pass=False,
                  pw_rigid=False,
                  shifts_opencv=True, 
                  nonneg_movie=True,
                  border_nan='copy', **normcorre_kwds):

    indices = tuple([slice(None) for _ in max_shifts])

    mc = MotionCorrect([memmap_name], dview=dview, max_shifts=max_shifts,
                  strides=strides, overlaps=overlaps, pw_rigid=pw_rigid,
                  max_deviation_rigid=max_deviation_rigid, 
                  shifts_opencv=shifts_opencv, nonneg_movie=True,
                  border_nan=border_nan, is3D=is3D, indices=indices, **normcorre_kwds)

    mc.motion_correct(save_movie=True)

    if pw_second_pass:
        print("Running second pass nonrigid motion correction...")
        mc.pw_rigid = True
        mc.template = mc.mmap_file
        mc.motion_correct(save_movie=True, template=mc.total_template_rig)

    return mc

def extract(mmapname, dview, n_processes=8, cnmf_second_pass=True, rf=15, stride=10, K=12, 
            gSig=(2, 2), merge_thresh = 0.8, p=0, fr=10, decay_time=1., gnb=2,
            min_SNR=3, rval_thr=0.7, use_cnn=False, **cnmf_kwds):
        # amounpl.it of overlap between the patches in pixels
        # number of neurons expected per patch
        # expected half size of neurons
        # merging threshold, max correlation allowed
         # half-size of the patches in pixels. rf=25, patches are 50x50
       # order of the autoregressive system)
    # approx final rate  (after eventual downsampling )
     # length of typical transient in seconds 
    # CNN classifier is designed for 2d (real) data
    # accept components with that peak-SNR or higher
     # accept components with space correlation threshold or higher


    Yr, dims, T = cm.load_memmap(mmapname)

    # Does this load the data into RAM???
    images = np.reshape(Yr.T, [T] + list(dims), order='C')    # reshape data in Python format (T x X x Y x Z)

    cnm = cnmf.CNMF(n_processes=n_processes, dview=dview, k=K, gSig=gSig, 
                stride=stride, gnb=gnb,  #rf=rf, # this parameter seems problematic 
                merge_thresh=merge_thresh, p=p, only_init_patch=True, **cnmf_kwds)

    cnm = cnm.fit(images)

    print(('Number of components:' + str(cnm.estimates.A.shape[-1])))

    # Evaluate components
    cnm.params.change_params(params_dict={'fr': fr,
                                          'decay_time': decay_time,
                                          'min_SNR': min_SNR,
                                          'rval_thr': rval_thr,
                                          'use_cnn': use_cnn})

    cnm.estimates.evaluate_components(images, cnm.params, dview=dview)
    print(('Keeping ' + str(len(cnm.estimates.idx_components)) +
            ' and discarding  ' + str(len(cnm.estimates.idx_components_bad))))
    
    if cnmf_second_pass:
        print("Second pass CNMF (seeded with first pass)...")
        cnm.params.set('temporal', {'p': p})
        cnm2 = cnm.refit(images)
        
        print("Computing DF/F")
        cnm2.estimates.detrend_df_f()

        return cnm2
    else:
        print("Computing DF/F")
        cnm.estimates.detrend_df_f()
        return cnm  



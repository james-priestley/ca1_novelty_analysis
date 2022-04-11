import os
import warnings

# Currently we can't limit multithreading directly in Suite2p, so we must set
# these environment variables prior to numpy/suite2p imports
N_PROCESSES = str(8)
os.environ["VECLIB_MAXIMUM_THREADS"] = N_PROCESSES
os.environ["NUMEXPR_NUM_THREADS"] = N_PROCESSES
os.environ["NUMBA_NUM_THREADS"] = N_PROCESSES
os.environ["NUMBA_DEFAULT_NUM_THREADS"] = N_PROCESSES
os.environ["OMP_NUM_THREADS"] = N_PROCESSES
os.environ["OPENBLAS_NUM_THREADS"] = N_PROCESSES
os.environ["MKL_NUM_THREADS"] = N_PROCESSES

OPS_0_6_16 = {

    # file paths
    'look_one_level_down': False,
    'fast_disk': [],
    'delete_bin': False,  # whether to delete binary file after processing
    'mesoscan': False,
    'h5py': [],
    'h5py_key': 'data',
    'save_path0': [],
    'save_folder': [],
    'subfolders': [],
    'bruker': False,

    # main settings
    'nplanes': 1,  # each tiff has these many planes in sequence
    'nchannels': 1,  # each tiff has these many channels per plane
    # this channel is used to extract functional ROIs (1-based)
    'functional_chan': 1,
    # calcium time constant -- this is the main parameter for deconvolution
    'tau':  0.7,
    # sampling rate -- for multi-plane imaging, this is the sampling rate per
    # complete stack/volume
    'fs': 10.,
    'force_sktiff': False,

    # output settings
    'preclassify': 0.,
    'save_mat': False,
    # combine multiple planes into a single result /single canvas for GUI
    # don't change this, or our import step will not work properly
    'combined': True,
    # um/pixels in X / um/pixels in Y (for correct aspect ratio in GUI)
    'aspect': 1.0,

    # bidirectional phase offset
    'do_bidiphase': False,
    'bidiphase': 0,

    # registration settings
    # whether to register data (2 forces re-registration)
    'do_registration': 1,
    'keep_movie_raw': False,
    # subsampled frames for finding reference image
    'nimg_init': 300,
    # number of frames per batch
    'batch_size': 2000,
    # max allowed registration shift, as a fraction of
    # frame max(width and height)
    'maxregshift': 0.1,
    # when multi-channel, you can align by non-functional channel (1-based)
    'align_by_chan': 1,
    'reg_tif': False,  # whether to save registered tiffs
    'reg_tif_chan2': False,  # whether to save channel 2 registered tiffs
    'subpixel': 10,  # precision of subpixel registration (1/subpixel steps)
    # ~1 good for 2P recordings, recommend >5 for 1P recordings
    'smooth_sigma': 1.15,
    # this parameter determines which frames to exclude when determining
    # cropping - set it smaller to exclude more frames
    'th_badframes': 1.0,
    'pad_fft': False,

    # non rigid registration settings
    'nonrigid': True,  # whether to use nonrigid registration
    # block size to register (** keep this a multiple of 2 **)
    'block_size': [128, 128],
    # if any nonrigid block is below this threshold, it gets smoothed until
    # above this threshold. 1.0 results in no smoothing
    'snr_thresh': 1.2,
    # maximum pixel shift allowed for nonrigid, relative to rigid
    'maxregshiftNR': 5,

    # 1P settings
    '1Preg': False,
    'spatial_hp': 25,
    'pre_smooth': 2,
    'spatial_taper': 50,

    # cell detection settings
    'roidetect': True,  # whether or not to run ROI extraction
    'sparse_mode': False,  # whether or not to run sparse_mode
    # use diameter for filtering and extracting (ignored for sparse_mode=True)
    'diameter': 12,
    # use spatial scale for sparse_mode (ignored for sparse_mode=False)
    # 0: multi-scale; 1: 6 pixels, 2: 12 pixels, 3: 24 pixels, 4: 48 pixels
    'spatial_scale': 1,
    # whether or not to keep ROIs fully connected (set to 0 for dendrites)
    'connected': True,
    # max number of binned frames for cell detection
    'nbinned': 10000,
    # maximum number of iterations to do cell detection
    'max_iterations': 20,
    # adjust the automatically determined threshold by this scalar multiplier
    'threshold_scaling': 1.0,
    # cells with more overlap than this get removed during triage, before
    # refinement
    'max_overlap': 0.75,
    # running mean subtraction with window of size 'high_pass'
    # (use low values for 1P)
    'high_pass': 100,

    # ROI extraction parameters
    # number of pixels to keep between ROI and neuropil donut
    'inner_neuropil_radius': 2,
    # minimum number of pixels in the neuropil
    'min_neuropil_pixels': 350,
    # pixels that are overlapping are thrown out (False)
    # or added to both ROIs (True)
    'allow_overlap': False,

    # channel 2 detection settings (stat[n]['chan2'], stat[n]['not_chan2'])
    'chan2_thres': 0.65,  # minimum for detection of brightness on channel 2

    # deconvolution settings
    'baseline': 'maximin',  # baselining mode (can also choose 'prctile')
    'win_baseline': 60.,  # window for maximin
    'sig_baseline': 10.,  # smoothing constant for gaussian filter
    'prctile_baseline': 8.,  # optional (whether to use a percentile baseline)
    'neucoeff': .7,  # neuropil coefficient
}

DB = {

    # these fields are generated by Suite2pImagingDataset objects automatically
    'h5py': [],
    'h5py_key': 'data',
    'look_one_level_down': False,
    'data_path': None,
    'subfolders': [],
    'fast_disk': None,
}

OPS_0_5_5 = {}


def get_ops(suite2p_version):
    """Returns the default options dictionary for Suite2p, specific to the
    suite2p version installed.

    Parameters
    ----------
    suite2p_version : str
        Suite2p version number. They don't include this under
        suite2p.__version__ for whatever reason, but you can access it in
        python via pkg_resources.get_distribution(suite2p)

    Returns
    -------
    ops : dict
        Default options dictionary
    """
    if suite2p_version == '0.6.16':
        return OPS_0_6_16
    elif suite2p_version == '0.5.5':
        return OPS_0_5_5
    elif suite2p_version == '0.7.5':
        return OPS_0_6_16
    else:
        warnings.warn("Untested suite2p version! Using 0.6.16 ops dictionary")
        return OPS_0_6_16

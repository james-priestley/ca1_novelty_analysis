import os
import shutil
import warnings

from abc import abstractmethod, ABCMeta
from sima import ImagingDataset
from pkg_resources import get_distribution

import matplotlib.pyplot as plt
import numpy as np
import psutil
from scipy.ndimage.filters import gaussian_filter
import sys
import pickle as pkl

import caiman as cm
import caiman.source_extraction.cnmf as cnmf
from caiman.components_evaluation import evaluate_components, estimate_components_quality_auto
from caiman.cluster import setup_cluster
from caiman.paths import caiman_datadir

from lab3.extraction.cnmf import convertToMemmap, AROIs, extract_signals, import_signals
from lab3.misc.sima_compatibility import sima_compatible
from lab3.misc.sima_sequences import _NoRMCorreRigidSequence

FASTSCRATCH = "/fastscratch/memmaps"

@sima_compatible
class CNMFImagingDataset(ImagingDataset):
    def extract(self, overwrite=False, reprocess=True, on_scratch=True, 
        first_obs=0, signal_channel='Ch2', n_processes=8, 
        # NormCorre arguments
        use_normcorre=True, max_shifts=(6, 6), strides=(48, 48), overlaps=(24, 24), 
        pw_second_pass=False, normcorre_kws={},
        # CNMF arguments
        cnmf_second_pass=True, rf=15, stride=10, K=12, gSig=(2, 2), gnb=2,
        merge_thresh=0.8, p=0, fr=10, decay_time=1., cnmf_kws={}):

        # TODO: document
        self.write_memmap(overwrite=overwrite, reprocess=reprocess, 
                            on_scratch=on_scratch, first_obs=first_obs, 
                            signal_channel=signal_channel, n_processes=n_processes)

        c, dview, n_processes = setup_cluster(
            backend='local', n_processes=n_processes, single_thread=False)

        try:
            if use_normcorre:
                self.motion_correct(dview, max_shifts=max_shifts, strides=strides, 
                        overlaps=overlaps, pw_second_pass=pw_second_pass, **normcorre_kws)

            print("Extracting with CNMF...")
            self.cnm = extract_signals.extract(self.mmapname, dview, n_processes=n_processes, 
                                                cnmf_second_pass=cnmf_second_pass, rf=rf, 
                                                stride=stride, K=K, gSig=gSig, gnb=gnb,
                                                merge_thresh=merge_thresh, 
                                                p=p, fr=fr, decay_time=decay_time, 
                                                **cnmf_kws)
            self.save_cnmf_results()
        except:
            raise
        finally:
            # Whatever happens, close the cluster
            cm.stop_server(dview=dview)

    def extract3D(self, max_shifts=(8, 8, 2), strides=(48, 48, 4), overlaps=(24, 24, 2), 
                    gSig=(2, 2, 2.5), **kwargs):
        """Identical to `extract`, just with default parameters set for 3D
        """

        self.extract(max_shifts=max_shifts, strides=strides, overlaps=overlaps, 
                    gSig=gSig, **kwargs)

    def write_memmap(self, overwrite=False, reprocess=True, on_scratch=True, 
                    first_obs=50, signal_channel='Ch2', n_processes=8):
        if on_scratch:
            self.mmapname = os.path.join(FASTSCRATCH, self.mmapname)

        # TODO: convert assumes sequence 0
        if not os.path.exists(self.mmapname) or overwrite:
            convertToMemmap.convert(self.savedir, self.mmapname, n_processes=n_processes,
                channel=signal_channel, first_obs=first_obs, n_frames=self.num_frames)
        elif reprocess:
            pass # use existing mmap
        else:
            raise OSError(f"`{self.mmapname}` already exists (reprocess=True to extract from "\
                            f"existing .mmap or overwrite=True to overwrite)")

    def motion_correct(self, dview, max_shifts=(6, 6), strides=(48, 48), 
                        overlaps=(24, 24), pw_second_pass=False, **normcorre_kws):

        is3D = self.frame_shape[0] > 1

        if is3D:
            assert len(max_shifts) == len(strides) == len(overlaps) == 3
            print("3D check passed")
        else:
            assert len(max_shifts) == len(strides) == len(overlaps) == 2

        print("Motion correcting...")
        self.mc = extract_signals.motion_correct(self.mmapname, dview=dview, 
                    max_shifts=max_shifts, strides=strides, overlaps=overlaps, 
                    pw_second_pass=pw_second_pass, is3D=is3D, **normcorre_kws)
        self.mmapname, = self.mc.mmap_file
        print("Done.")
        print(f"MC file: {self.mmapname}")

    def import_results(self, **kwargs):
        """Import Suite2p signal results using the new-style signal formats
        (dataframes and h5 stores).

        Parameters
        ----------
        ds : CNMFImagingDataset
            Instance of a Suite2pStrategy, which has been previously extracted.
        label : str, optional
            ROI label to store the signals and ROI masks under. Defaults to
            'cnmf'.
        channel : {str or int}, optional
            Name or index of dynamic channel in the imaging dataset. Defaults
            to 'Ch2'.
        overwrite : bool, optional
            If key '/{channel}/{label}' already exists in the dataset signals
            file, overwrite with new import. Note this will delete any
            additional signal types in the store in addition to the 'raw' and
            'npil' (e.g. 'dfof'), to prevent mixtures of signals derived from
            different imports. Defaults to False, which will raise an error if
            the desired key already exists.

        See also
        --------
        lab3.extraction.import_signals.import_to_signals_file
        """

        import_signals.import_to_signals_file(self, **kwargs)

    def apply_mc_results(self, write=False):
        """Apply displacements calculated by NormCorre to the underlying dataset.
        Currently only supports rigid motion correction.

        Parameters
        ----------
        self : CNMFDataset
            Instance of a CNMFDataset, which has been previously extracted.
        overwrite : bool, optional
            Whether to save the motion corrected dataset as a sima directory.
        """

        new_sequences = []

        for seq in self.sequences:
            mc_seq = _NoRMCorreRigidSequence(seq, self.mc)
            new_sequences.append(mc_seq)

        self.sequences = new_sequences

        if write:
            # backup_dir(self.savedir, delete_original=False)
            self.save()

    def save_cnmf_results(self):
        cnmf_pkl = os.path.join(self.savedir, 'cnm.pkl')

        # Make pickleable
        self.cnm.dview = None 
        self.mc.dview = None 

        with open(cnmf_pkl, "wb") as cnmf_file:
            print("Saving cnmf results...")
            pkl.dump(self.cnm, cnmf_file)

        if hasattr(self, 'mc'):
            print("Saving mc results...")
            mc_pkl = os.path.join(self.savedir, 'normcorre_mc.pkl')
            with open(mc_pkl, "wb") as normcorre_file:
                pkl.dump(self.mc, normcorre_file)

    @property 
    def dims(self):
        return self.cnm.dims

    @property
    def cnm(self):
        try:
            return self._cnm
        except AttributeError:
            cnmf_pkl = os.path.join(self.savedir, 'cnm.pkl')
            with open(cnmf_pkl, "rb") as cnmf_file:
                self._cnm = pkl.load(cnmf_file)
            return self._cnm 

    @cnm.setter 
    def cnm(self, value):
        self._cnm = value

    @property
    def mc(self):
        try:
            return self._mc
        except AttributeError:
            mc_pkl = os.path.join(self.savedir, 'normcorre_mc.pkl')
            with open(mc_pkl, "rb") as normcorre_file:
                self._mc = pkl.load(normcorre_file)
            return self._mc
    
    @mc.setter 
    def mc(self, value):
        self._mc = value

    @property
    def mmapname(self):
        try:
            return self._mmapname
        except AttributeError:
            z, y, x, c = self.frame_shape
            t = self.num_frames
            mmapname = f"Yr_d1_{y}_d2_{x}_d3_{z}_order_C_frames_{t}_.mmap"
            self._mmapname = mmapname
            return mmapname

    @mmapname.setter 
    def mmapname(self, value):
        self._mmapname = value
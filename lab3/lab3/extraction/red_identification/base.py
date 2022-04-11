import sima
import shutil
import os
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.collections import PolyCollection

from lab3.experiment import ImagingOnlyExperiment, utils
from lab3.extraction.s2p.batch import BatchSuite2pImagingDataset

class PostHocIdentificationDataset(BatchSuite2pImagingDataset):
    def __init__(self, functional_sessions, savedir, static_session=None, static_tiff=None):
        child_dirs = []
        
        if isinstance(functional_sessions, list):
            child_dirs.extend(functional_sessions)
        elif isinstance(functional_sessions, str):
            child_dirs.append(functional_sessions)
        elif functional_sessions is None:
            # Load from existing directory
            super().__init__(None, savedir)
            self.static_savedir = self.child_dirs[-1]
            return
        else:
            raise ValueError(f"Invalid sessions {functional_sessions}")
            
        if static_session is not None:
            self.static_savedir = static_session
            child_dirs.append(static_session)
        elif static_tiff is not None:
            tiff_seq = sima.Sequence.create("TIFF", path=static_tiff)
            
            frame = tiff_seq._get_frame(0)
            
            ndarray_seq = sima.Sequence.create("ndarray", frame.reshape((1,) + frame.shape))
            
            self.static_savedir = static_tiff.replace('.tif', '.sima')
            
            if os.path.exists(self.static_savedir):
                shutil.rmtree(self.static_savedir)
            
            static_ds = sima.ImagingDataset([ndarray_seq], 
                                            savedir=self.static_savedir, 
                                            channel_names=['Ch2']) # intentionally fictitious
            child_dirs.append(self.static_savedir)
        else:
            raise ValueError("Either `static_session` or `static_tiff` must be specified!")

        if os.path.exists(savedir):
            shutil.rmtree(savedir)
        
        super().__init__(child_dirs=child_dirs, savedir=savedir)
        
    def import_red_rois(self, threshold=None, threshold_pct=99., green_label='PYR', red_label='red', 
                        current_label='suite2p', 
                        plot_summary=False, **kwargs):
        """
        
        Run this *after* doing `import_results_to_children`
        
        """
        static_expt = ImagingOnlyExperiment(self.static_savedir)
        signals = static_expt.signals(label=current_label, signal_type='raw').mean(axis=1)

        if not threshold:
            threshold = np.percentile(static_expt.imaging_dataset.time_averages, threshold_pct)

        new_label_rois = signals.loc[signals > threshold].index
        
        for sima_path in self.child_dirs:
            expt = ImagingOnlyExperiment(sima_path)
            utils.split_signals(expt, new_label=red_label, 
                                new_label_rois=new_label_rois, 
                                current_label=current_label, 
                                **kwargs)

            utils.split_signals(expt, new_label=green_label, 
                    new_label_rois=new_label_rois, 
                    current_label=current_label, 
                    inverse=True,
                    **kwargs)
            
        if plot_summary:
            self.plot_summary(green_label=green_label, red_label=red_label)
        
    def plot_summary(self, green_label=None, red_label=None):
        static_expt = ImagingOnlyExperiment(self.static_savedir)
        img = static_expt.imaging_dataset.time_averages[0,...,0]
        
        pct1 = np.nanpercentile(img.flatten(), 1)
        pct99 = np.nanpercentile(img.flatten(), 98)
        
        rois = static_expt.roi_list(label=green_label)
        
        if red_label is not None:
            red_rois = static_expt.roi_list(label=red_label)
        else:
            red_rois = []
            
        with PdfPages(os.path.join(self.savedir, 'extraction_summary.pdf')) as pdf:
            fig, ax = plt.subplots(figsize=(10,10))
            ax.imshow(img, cmap='gray', vmin=pct1, vmax=pct99, aspect='equal')
            
            all_roi_verts = []
            red_verts = []
            
            for r in rois:
                all_roi_verts.append(r.coords[0][:,:2])
                
            for r in red_rois:
                red_verts.append(r.coords[0][:,:2])
            
            collL = PolyCollection(all_roi_verts,alpha=0.5)
            collR = PolyCollection(red_verts,alpha=0.5, color='r')
            ax.add_collection(collL)
            ax.add_collection(collR)

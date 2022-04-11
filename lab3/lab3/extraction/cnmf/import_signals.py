import os

import sima
import pandas as pd 
import numpy as np
from scipy.ndimage import center_of_mass

from lab3.signal import SignalFile
from lab3.extraction.cnmf import AROIs

def import_to_signals_file(ds, label='cnmf', channel='Ch2',
                           overwrite=False):
    """Import CNMF signal results using the new-style signal formats
    (dataframes and h5 stores).

    Parameters
    ----------
    ds : CNMFDataset
        Instance of a CNMFDataset, which has been previously extracted.
    label : str, optional
        ROI label to store the signals and ROI masks under. Defaults to
        'cnmf'.
    channel : {str or int}, optional
        Name or index of dynamic channel in the imaging dataset. Defaults to
        'Ch2'.
    overwrite : bool, optional
        If key '/{channel}/{label}' already exists in the dataset signals file,
        overwrite with new import. Note this will delete any additional signal
        types in the store in addition to the 'raw' and 'npil' (e.g. 'dfof'),
        to prevent mixtures of signals derived from different imports. Defaults
        to False, which will raise an error if the desired key already exists.
    """

    # check if key already exists in signals file, and deal with overwrites
    with SignalFile(os.path.join(ds.savedir, "signals.h5")) as signal_file:
        base_key = f"/{channel}/{label}"
        if base_key in signal_file:
            assert overwrite, f"Signals already exist for {base_key}. " \
                + "Set overwrite=True to re-import.\nCurrent file structure:" \
                + f" \n {signal_file.info()}"

            print(f"Deleting previous import at {base_key}")
            signal_file.remove(base_key)

        # create a sima ROI list and save to the underlying imaging dataset
        print("Making rois...")
        rois = _make_cell_rois(ds)
        print(f"Made {len(rois)} rois")
        ds.add_ROIs(rois, label=label)

        background_rois = _make_background_rois(ds)
        print(f"Made {len(background_rois)} background components")
        ds.add_ROIs(background_rois, label='background')

        # create signal dataframes
        # TODO replace this with proper signal classes for each type,
        # and store metadata
        idx = pd.Index([roi.label for roi in rois], name='roi_label')
        bkd_idx = pd.Index([roi.label for roi in background_rois], 
                            name='roi_label')

        raw = pd.DataFrame(ds.cnm.estimates.C, index=idx) 
        dfof = pd.DataFrame(ds.cnm.estimates.F_dff, index=idx)
        background = pd.DataFrame(ds.cnm.estimates.f, index=bkd_idx)

        if ds.cnm.params.temporal['p'] > 0:
            spikes = pd.DataFrame(ds.cnm.estimates.S, index=idx)

        # store signals
        print("Storing...")
        signal_file.put(f"{base_key}/raw", raw)
        signal_file.put(f"{base_key}/dfof", dfof)        
        signal_file.put(f"{base_key}/background", background)
        if ds.cnm.params.temporal['p'] > 0:
            signal_file.put(f"{base_key}/spikes", spikes)

        print(signal_file.info())

def name_rois(arois):
    names = []
    for r in arois:
        com = center_of_mass(r.tensor)
        hashes = [f"{hash(coord)%10**4:04}" for coord in com]
        name = '-'.join(hashes)
        names.append(name)
    return names

def _make_cell_rois(ds):
    arois = AROIs.AROIs(matrix=ds.cnm.estimates.A, dims=ds.dims)
    roi_labels = name_rois(arois)
    arois = AROIs.AROIs(matrix=arois.matrix, labels=roi_labels, dims=ds.dims)
    rois = arois.roilist

    return rois

def _make_background_rois(ds):
    num_background = ds.cnm.params.temporal['nb']
    background_labels = [f"BACKGROUND{i}" for i in range(num_background)]
    background_arois = AROIs.AROIs(matrix=ds.cnm.estimates.b, 
                                    dims=ds.dims, labels=background_labels)
    background_rois = background_arois.roilist

    return background_rois

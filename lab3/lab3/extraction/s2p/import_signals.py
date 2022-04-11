import os

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from sima.ROI import ROIList, ROI

from lab3.signal import SignalFile
from lab3.misc.progressbar import ProgressBar


def import_to_signals_file(ds, label='suite2p', channel='Ch2',
                           overwrite=False):
    """Import Suite2p signal results using the new-style signal formats
    (dataframes and h5 stores).

    Parameters
    ----------
    ds : Suite2pStrategy
        Instance of a Suite2pStrategy, which has been previously extracted.
    label : str, optional
        ROI label to store the signals and ROI masks under. Defaults to
        'suite2p'.
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
        rois = _make_roi_masks(ds)
        print(f"Made {len(rois)} rois")
        ds.add_ROIs(rois, label=label)

        # create signal dataframes
        # TODO replace this with proper signal classes for each type,
        # and store metadata
        idx = pd.Index([roi.label for roi in rois], name='roi_label')
        raw = pd.DataFrame(ds.raw_signals[ds.cell_indicator], index=idx)
        npil = pd.DataFrame(ds.npil_signals[ds.cell_indicator], index=idx)

        try:
            raw.loc[:, ds.bad_frames] = np.nan
            npil.loc[:, ds.bad_frames] = np.nan
            print(f"Masked frames {ds.bad_frames}")
        except FileNotFoundError:
            pass

        # Handle a suite2p quirk - 0 means 'nan'
        raw[raw == 0] = np.nan
        npil[npil == 0] = np.nan

        # store signals
        # timestamp = datetime.strftime(datetime.now(), '%Y-%m-%d-%Hh%Mm%Ss')
        print("Storing...")
        signal_file.put(f"{base_key}/raw", raw)
        signal_file.put(f"{base_key}/npil", npil)
        print(signal_file.info())


def _make_roi_masks(ds):
    """Parse Suite2p results files to create an ROI object for each cell

    Parameters
    ----------
    ds : Suite2pStrategy
        Instance of a Suite2pStragegy, which has been previously extracted.

    Returns
    -------
    rois : sima.ROI.ROIList
    """

    num_planes, num_y, num_x = ds.sequences[0].shape[1:-1]

    # get row index of identified cells. note this is the position of each cell
    # in the total matrix of signals (combined across planes)
    cell_idx = ds.cell_indicator

    rois = [] * len(cell_idx)
    cum_rois = 0
    blank_mask = [coo_matrix((num_y, num_x)) for z in range(num_planes)]
    claimed_identifiers = []
    for z in range(len(ds.ops_list)):
        # get ROI information for current plane
        print(f"Plane {z}")
        stat = ds.get_stat_file(plane=z)

        # get the subset of cell_idx that appear in the current plane,
        # and shift the indices to match those in the plane stat file
        plane_idx = (cell_idx >= cum_rois) \
            & (cell_idx < (cum_rois + len(stat)))
        plane_idx = cell_idx[plane_idx] - cum_rois

        # get info for each valid ROI from stat and create mask
        p = ProgressBar(len(stat[plane_idx]))
        for i, sdict in enumerate(stat[plane_idx]):
            mask = list(blank_mask)
            mask[z] = coo_matrix((sdict['lam'], (sdict['ypix'],
                                  sdict['xpix'])),
                                 shape=(num_y, num_x))
            identifier = '{0:04d}-{1:04d}-{2:04d}'.format(
                z, int(np.mean(sdict['ypix'])),
                int(np.mean(sdict['xpix'])))
            if identifier in claimed_identifiers:
                # augment identifier with a number to denote the nth ROI that
                # shares the same centroid
                num_shared_centroids = np.sum(
                    [identifier in i for i in claimed_identifiers])
                identifier += f'_{num_shared_centroids}'

            # add identifier to claimed list
            claimed_identifiers.append(identifier)

            roi = ROI(mask=mask, label=identifier, id=identifier)
            rois.append(roi)
            p.update(i)
        print(' Done!')
        cum_rois += len(stat)

    return ROIList(rois)


def import_to_pkl():
    """For old-style imports"""
    raise NotImplementedError

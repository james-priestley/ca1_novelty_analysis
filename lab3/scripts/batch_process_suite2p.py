# This script should
# - In "extract" mode
#     - Build a list of sima datasets to batch process
#     - Create a new concatenated sima dataset
#     - Cast the new dataset as a Suite2pImagingDataset object
#     - Run Suite2p extraction
#     - Optionally apply motion correction results to the batch and/or original
#       datasets, as sequence wrappers
#
# - In "import" mode
#     - Cut up suite2p results according to the length of the original
#     - sequences
#     - Create signals.h5 entries in the original imaging dataset objects

import os
import numpy as np
import argparse

from lab3.extraction.s2p import Suite2pImagingDataset
from lab3.extraction.s2p.batch import BatchSuite2pImagingDataset

def check_kwds(path, kwds):
    if kwds is None:
        return True
    else:
        word_in_path = [word in path for word in kwds]
        return np.all(word_in_path)

def get_sima_paths(paths, keywords):
    sima_paths = []
    for path in paths:
        for dirpath, _, _ in os.walk(path):
            if dirpath.endswith('.sima') and check_kwds(dirpath, keywords):
                sima_paths.append(dirpath)

    return sorted(sima_paths)

def extract_separate(sima_paths, args):
    for sima_path in sima_paths:
        s2p_ds = Suite2pImagingDataset.load(sima_path)
        ops_kws = {}

        if args.nonrigid:
            ops_kws["nonrigid"] = True
        else:
            ops_kws["nonrigid"] = False 

        s2p_ds.extract(signal_channel=args.signal_channel, 
                        fs=args.fs, 
                        n_processes=args.n_processes,
                        overwrite=args.overwrite, 
                        reprocess=args.reprocess, 
                        register=args.register, 
                        collapse_z=args.collapse_z,
                        ops_kws=ops_kws)

def import_separate(sima_paths, args):
    for sima_path in sima_paths:
        s2p_ds = Suite2pImagingDataset.load(sima_path)
        if args.apply_mc:
            s2p_ds.apply_mc_results(overwrite=True)
        s2p_ds.import_results(label=args.label, channel=args.signal_channel, 
                            overwrite=True)

def extract_batch(sima_paths, batch_path, args):

    batch_ds = BatchSuite2pImagingDataset(child_dirs=sima_paths, savedir=batch_path)
    ops_kws = {}

    if args.nonrigid:
        ops_kws["nonrigid"] = True
    else:
        ops_kws["nonrigid"] = False 

    batch_ds.extract(signal_channel=args.signal_channel, 
                    fs=args.fs, 
                    n_processes=args.n_processes,
                    overwrite=args.overwrite, 
                    reprocess=args.reprocess, 
                    register=args.register, 
                    collapse_z=args.collapse_z,
                    ops_kws=ops_kws)

def import_batch(batch_path, args):
    batch_ds = BatchSuite2pImagingDataset.load(batch_path)
    if args.apply_mc:
        batch_ds.apply_mc_to_children(overwrite=True)
    batch_ds.import_results_to_children(label=args.label, channel=args.signal_channel, 
                        overwrite=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                description="Performs (batch) motion correction and extraction with suite2p. " 
                            "Most important parameters in ALL CAPS")
    parser.add_argument("paths", nargs='+', type=str, 
            help='list of simas, or directories to look for simas')
    parser.add_argument("--mode", "-m", type=str, default="unified", 
            help='`extract` or `import` or `unified` (default)')
    parser.add_argument("--keywords", "-k", nargs='+', type=str, 
            help='list of keywords that must be in each sima path (optional)')
    parser.add_argument("--dry_run", action="store_true", 
            help='print simas to extract and quit')
    parser.add_argument("--batch", "-b", action="store_true", 
            help='EXTRACT ALL SIMAS AS BATCH (default: extract one-by-one)')
    parser.add_argument("--reprocess", "-p", action="store_true", 
            help='reprocess existing binaries')
    parser.add_argument("--overwrite", "-o", action="store_true", 
            help='overwrite existing binaries')
    parser.add_argument("--register", "-r", action="store_true", 
            help='WHETHER TO PERFORM MOTION CORRECTION')
    parser.add_argument("--collapse_z", action="store_true", 
            help='collapse planes to single plane')
    parser.add_argument("--nonrigid", "-n", action="store_true", 
            help='use nonrigid motion correction')
    parser.add_argument("--signal_channel", "-c", type=str, default="Ch2", 
            help='channel to extract')
    parser.add_argument('--n_processes', default=8, type=int, 
            help='number of processes to use')
    parser.add_argument("--apply_mc", action="store_true",
            help="import motion correction to original sequence")
    parser.add_argument("--fs", "-f", type=float, default=30, 
            help='FRAME RATE OF SESSIONS (default 30)')
    parser.add_argument("--label", "-l", type=str, default="suite2p",
            help="LABEL OF IMPORTED SIGNALS (default suite2p)")

    args = parser.parse_args()
    
    sima_paths = get_sima_paths(args.paths, args.keywords)
    batch_path = os.path.join(args.paths[0], "batch_s2p.sima")

    if args.dry_run:
        print(sima_paths)
    else:
        if args.mode != "import":
            if args.batch:
                extract_batch(sima_paths, batch_path, args)
            else:
                extract_separate(sima_paths, args)

        if args.mode != "extract":
            if args.batch:
                import_batch(batch_path, args)
            else:
                import_separate(sima_paths, args)
            




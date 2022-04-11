import os
import numpy as np
import argparse

import lab3.misc.script_helpers as helpers
from lab3.experiment.convert import OldImagingExperiment, unpickle_old

def convert_single_expt(sima_path, labels=("*",), channel='Ch2'):
    old_expt = OldImagingExperiment(sima_path)

    signal_types = ['raw', 'dfof']

    if "*" in labels:
        ch_id = old_expt.imaging_dataset._resolve_channel(channel)

        signals_file_path = os.path.join(sima_path, f'signals_{ch_id}.pkl')
        data = unpickle_old(signals_file_path)
        labels = data.keys()

    for label in labels:
        print(f"Found label {label}")
        for signal_type in signal_types:
            try:
                old_expt.write_newstyle(signal_type=signal_type, 
                                    label=label, 
                                    channel=channel)
                print(f"Converted {signal_type} signals for label '{label}'")
            except:
                pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                description="Converts data saved in the old style to new-style")
    parser.add_argument("paths", nargs='+', type=str, 
            help='list of simas, or directories to look for simas')
    parser.add_argument("--keywords", "-k", nargs='+', type=str, 
            help='list of keywords that must be in each sima path (optional)')
    parser.add_argument("--channel", '-c', type=str, default='Ch2', 
            help='channel of signals to convert (default: Ch2)')    
    parser.add_argument("--dry_run", action="store_true", 
            help='print simas to extract and quit')
    parser.add_argument("--labels", "-l", nargs='+', type=str, default=("*",),
            help="labels of signals to convert (default: all)")

    args = parser.parse_args()

    sima_paths = helpers.get_sima_paths(args.paths, keywords=args.keywords)

    for sima_path in sima_paths:
        print(f"Converting {sima_path}...")

        if not args.dry_run:
            try:
                convert_single_expt(sima_path, 
                                labels=args.labels, 
                                channel=args.channel)
            except Exception as exc:
                print(f"Failed on {sima_path} because: `{exc}`")

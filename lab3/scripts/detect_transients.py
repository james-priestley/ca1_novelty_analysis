import sys
import re
import os
import argparse
from datetime import datetime
import matplotlib as mpl
# mpl.use('pdf')
import matplotlib.pyplot as plt
# from matplotlib.backends.backend_pdf import PdfPages

import numpy as np
import pandas as pd
import datetime
import pickle as pkl
import json
import numpy.ma as ma
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
# try:
#     from bottleneck import nanstd
# except ImportError:
#     from numpy import nanstd
# nanstd above was never being used
# import cPickle as pickle
import itertools as it
import warnings

from lab3.experiment.base import BehaviorExperiment, ImagingExperiment, fetch_trials
from lab3.experiment.group import ExperimentGroup

from lab3.event.transient import Transient, SingleCellTransients, ExperimentTransientGroup

import lab3.signal.transients as trans

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                description="Detects transients (BETA)")
    parser.add_argument("project_name", type=str, 
            help='database to search for experiments in')
    parser.add_argument("--mouse", "-m", nargs='+', type=str, 
            help='mice to process')
    parser.add_argument("--keywords", "-k", nargs='+', type=str, 
            help='list of keywords that must be in each sima path (optional)')
    parser.add_argument("--trials", "-t", nargs='+', type=str, 
            help='list of trials to detect transients on immediately')
    parser.add_argument("--channel", '-c', type=str, default='Ch2', 
            help='channel of signals to convert (default: Ch2)')    
    parser.add_argument("--dry_run", action="store_true", 
            help='print simas to detect and quit')
    parser.add_argument("--label", "-l", type=str, default="suite2p",
            help="labels of signals to detect transients on (default: 'suite2p')")

    args = parser.parse_args()

    if args.trials:
        trials = args.trials 
    else: 
        trials = fetch_trials(project_name=args.project_name, mouse_name=args.mouse)
    
    expts = ExperimentGroup.from_trial_ids(trials)

    if args.keywords:
        for kwd in args.keywords:
            expts = ExperimentGroup([e for e in expts if kwd in e.sima_path])

    for expt in expts:
        print(f"Detecting transients for {expt}...")
        try:
            if not args.dry_run:
                thresholds, noise = trans.calculate_transient_thresholds([expt], label=args.label,
                                                                    channel=args.channel)
                transients = trans.identify_transients([expt], thresholds, noise=noise, 
                                                label=args.label, channel=args.channel, 
                                                demixed=False, save=True)
        except Exception as exc:
            print(f"Failed on {expt} with {exc}")
            continue

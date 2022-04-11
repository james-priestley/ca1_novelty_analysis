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
from numpy import nanstd
# import cPickle as pickle
import itertools as it
import warnings

from lab3.experiment.base import BehaviorExperiment, ImagingExperiment, fetch_trials
from lab3.experiment.group import ExperimentGroup

from lab3.event.transient import Transient, SingleCellTransients, ExperimentTransientGroup
from lab3.filters import EventFilter

# setting up gobal variables for calculating transients
class configs:
    if (os.path.exists('configs.json')):
        settings = json.load(
            open('configs.json')).get('calculateTransients', {})
    else:
        settings = {}

    # keep first value .01 for initial pruning and baseline recalculation
    P_VALUES = settings.get('P_VALUES', [.01, .05])

    ##
    # Transient identification parameters
    ##
    ON_THRESHOLD = settings.get('ON_THRESHOLD', 2)  # threshold for starting an event (in sigmas)
    OFF_THRESHOLD = settings.get('OFF_THRESHOLD', 0.5)  # end of event (in sigmas)
    # for which to calculate parameters (all events greater are in final bin)
    MAX_SIGMA = settings.get('MAX_SIGMA', 5)
    MIN_DURATION = settings.get('MIN_DURATION', 0.1) # for which to calculate parameters
    MAX_DURATION = settings.get('MAX_DURATION', 5) # for which to calculate parameters
    N_BINS_PER_SIGMA = settings.get('N_BINS_PER_SIGMA', 2)
    N_BINS_PER_SEC = settings.get('N_BINS_PER_SEC', 4)
    # to iteratively identify events and re-estimate noise
    N_ITERATIONS = settings.get('N_ITERATIONS', 3)

    parameters = {
        'on_threshold': ON_THRESHOLD, 'off_threshold': OFF_THRESHOLD,
        'max_sigma': MAX_SIGMA, 'min_duration': MIN_DURATION,
        'max_duration': MAX_DURATION, 'n_bins_per_sigma': N_BINS_PER_SIGMA,
        'n_bins_per_sec': N_BINS_PER_SEC, 'n_iterations': N_ITERATIONS}

    parameters_text = 'ON_THRESHOLD = %.2f, OFF_THRESHOLD = %.2f, \
                       MAX_SIGMA = %.2f, MIN_DURATION = %.2f, \
                       MAX_DURATION = %.2f, N_BINS_PER_SIGMA = %d, \
                       N_BINS_PER_SEC = %d, N_ITERATIONS = %d' % \
        (float(ON_THRESHOLD), float(OFF_THRESHOLD), float(MAX_SIGMA),
         float(MIN_DURATION), float(MAX_DURATION), N_BINS_PER_SIGMA,
         N_BINS_PER_SEC, N_ITERATIONS)

    # one extra bin for events > the max time/sigma
    # parameters above must be chosen such that nBins are integers

    nTimeBins = int((MAX_DURATION - MIN_DURATION) * N_BINS_PER_SEC + 1)
    nSigmaBins = int((MAX_SIGMA - ON_THRESHOLD) * N_BINS_PER_SIGMA + 1)

def estimate_noise(expt, exclude_transients=False, channel='Ch2', label=None,
                   demixed=False):
    """
    Estimate the noise of each cell in the experiment.  If no transients have
    been identified, the noise is coarsely estimated to be the std of the trace
    (concatenated across cycles), a valid assumption for infrequently spiking
    pyramidal cells.  If transients are available, these epochs are excluded
    from the concatenated traces before calculating the std

    exclude_transients -- a transients structure (np record array)
    """
    
    signals = expt.signals(signal_type='dfof', label=label, channel=channel)
    imData = signals.to_numpy()[...,None]
    nCycles = imData.shape[2]
    # start_idx = expt.imagingIndex(10)
    # end_idx = expt.imagingIndex(45) - 1
    # imData = imData[:, start_idx:end_idx]
    concatenated_data = imData[:, :, 0]
    for i in range(nCycles - 1):
        concatenated_data = np.concatenate(
            (concatenated_data, imData[:, :, i + 1]), axis=1)

    if exclude_transients is True:

        expt_transients = expt.events(signal_type='imaging', event_type='transients', 
            label=label, channel=channel)

        activity = np.zeros(imData.shape, 'bool')

        if activity.ndim == 2:
            activity = activity.reshape(activity.shape[0],
                                        activity.shape[1], 1)

        for cell in expt_transients:
            roi_label  = cell.name
            cell_index = signals.level['roi_label'].get_loc(roi_label)

            for transient in cell:
                start = int(transient.onset)
                end = int(transient.offset)
                activity[cell_index, start:end + 1, 0] = True

        concatenated_activity = activity[:, :, 0]
        for i in range(nCycles - 1):
            concatenated_activity = np.concatenate(
                (concatenated_activity, activity[:, :, i + 1]), axis=1)

        # nan mask
        nan_mask = np.zeros(concatenated_activity.shape, dtype=bool)
        nan_mask[np.where(np.isnan(concatenated_data))] = True
        concatenated_activity = concatenated_activity | nan_mask

        masked_imData = ma.array(concatenated_data, mask=concatenated_activity)
        noise = masked_imData.std(axis=1).data

    else:
        noise = nanstd(concatenated_data, axis=1)


    return noise

def identify_events(data):
    # generator function which yields the start and stop frames for putative
    # events (data[start, stop] yields the full event)

    # accepts data in terms of sigmas
    L = len(data)
    start_index = 0

    while start_index < L:
        starts = np.where(data[start_index:] > configs.ON_THRESHOLD)[0].tolist()
        if starts:
            # start is the frame right before it crosses ON_THRESHOLD
            # (inclusive)
            abs_start = np.max([0, start_index + starts[0] - 1])
        else:
            break

        ends = np.where(data[abs_start + 1:] < configs.OFF_THRESHOLD)[0].tolist()
        if ends:
            # end is the first frame after it crosses OFF_THRESHOLD (event is
            # inclusive of abs_end -- need to add 1 to include this frame when
            # slicing)
            abs_end = abs_start + ends[0]
        else:
            break

        start_index = abs_end + 1

        yield abs_start, abs_end
#         wrap this with event constructor. lab3.events
#         get rid of yield and return an event group instead

def add_events_to_histogram(data, sigma, frame_period, direction, counter):
    """
    Given a timeseries (data), identify putative events and add them to counter
    """

    if direction is 'negative':
        data = -1 * data / sigma
    else:
        data = data / sigma

    for abs_start, abs_end in identify_events(data):

        dur = (abs_end - abs_start) * frame_period
        amp = np.nanmax(data[abs_start:abs_end + 1])

        if np.isnan(amp):
            continue

        if dur >= configs.MIN_DURATION:
            # bins are of the form [start, end)
            # add one for slicing
            sigma_bin_ind = int(np.floor(configs.N_BINS_PER_SIGMA * amp) -
                                configs.ON_THRESHOLD * configs.N_BINS_PER_SIGMA + 1)
            if sigma_bin_ind > configs.nSigmaBins:
                sigma_bin_ind = configs.nSigmaBins
            # add one for slicing
            time_bin_ind = int(np.floor(configs.N_BINS_PER_SEC * dur) -
                               configs.MIN_DURATION * configs.N_BINS_PER_SEC + 1)
            if time_bin_ind > configs.nTimeBins:
                time_bin_ind = configs.nTimeBins

            counter[:sigma_bin_ind, :
                    time_bin_ind] += np.ones([sigma_bin_ind, time_bin_ind])

    return counter

def calculate_event_histograms(experimentList, exclude_transients=False,
                               channel='Ch2', label=None, demixed=False):
    """
    Recursively calls add_events_to_histogram in order to pool event across
    cells across experiments
    """

    negative_event_counter = np.zeros((configs.nSigmaBins, configs.nTimeBins))
    positive_event_counter = np.zeros((configs.nSigmaBins, configs.nTimeBins))
    noise_dict = {}

    for expt in experimentList:
        # if exclude_transients is None:
        #     exclusion = None
        # else:
        #     exclusion = exclude_transients[expt]

#         this method is not implemented in lab3 so i decided to comment it out.

        # valid_filter = expt.validROIs(
        #     fraction_isnans_threshold=0, contiguous_isnans_threshold=0,
        #     dFOverF='from_file', channel=channel, label=label, demixed=demixed)
        valid_filter = None

        frame_period = expt.frame_period

        signals = expt.signals(signal_type='dfof', label=label, channel=channel)
        imData = signals.to_numpy()[...,None]

        noise_dict[expt.trial_id] = estimate_noise(expt, exclude_transients=exclude_transients,
                                          channel=channel, label=label,
                                          demixed=demixed)

#         valid_indices = expt._filter_indices(
#             valid_filter, channel=channel, label=label)

        for cell_data, sigma in zip(imData, noise_dict[expt.trial_id]): #[valid_indices]):
            for cycle_data in cell_data.T:
                positive_event_counter = add_events_to_histogram(
                    cycle_data, sigma, frame_period, 'positive',
                    positive_event_counter)
                negative_event_counter = add_events_to_histogram(
                    cycle_data, sigma, frame_period, 'negative',
                    negative_event_counter)

    return positive_event_counter, negative_event_counter, noise_dict


def calculate_transient_thresholds(experimentList, p=[.05], return_figs=False,
                                   fit_type='pw_linear',
                                   exclude_transients=False, channel='Ch2',
                                   label=None, demixed=False):

    if fit_type is None:
        fit_type = 'pw_linear'

    positive_events, negative_events, noise = calculate_event_histograms(
        experimentList, exclude_transients=exclude_transients, channel=channel,
        label=label, demixed=demixed)

    fpr = negative_events / positive_events

    d = np.linspace(configs.MIN_DURATION, configs.MAX_DURATION, configs.nTimeBins)
    d_fit_axis = np.linspace(configs.MIN_DURATION, configs.MAX_DURATION, 10000)
    thresholds = np.empty([configs.nSigmaBins, len(p)])

    if return_figs:
        nRows = 2
        nCols = 2
        nHistFigs = int(np.ceil(configs.nSigmaBins / float(nRows * nCols)))

        figs = [[] for x in range(nHistFigs)]
        axs = []
        fpr_fig, fpr_ax = plt.subplots(1, 1, figsize=(15, 8))

        for fig in xrange(nHistFigs):
            figs[fig], ax = plt.subplots(nRows, nCols, figsize=(15, 8))
            if ax.ndim == 1:
                ax = ax[:, np.newaxis]
            axs = np.hstack([axs, ax.flatten()])

        colors = plotting.color_cycle()
        color_list = []
        for sigma_ind in range(configs.nSigmaBins):
            ax = axs[sigma_ind]
            ax.bar(d, positive_events[sigma_ind],
                   width=1. / configs.N_BINS_PER_SEC, color='b')

            ax.bar(-1 * d - 1. / configs.N_BINS_PER_SEC,
                   negative_events[sigma_ind], width=1. / configs.N_BINS_PER_SEC,
                   color='r')

            ax.set_xticks(np.arange(-configs.MAX_DURATION, configs.MAX_DURATION + 1, 1))
            ax.set_xticklabels(np.arange(-configs.MAX_DURATION, configs.MAX_DURATION + 1, 1))

            ax.set_xlabel('Duration (sec)')
            ax.set_ylabel('Number of events')
            sigma_level = np.around(
                configs.ON_THRESHOLD + sigma_ind * (1. / configs.N_BINS_PER_SIGMA), decimals=3)
            ax.set_title('>= %s Sigma' % str(sigma_level))

            c = colors.next()
            fpr_ax.plot(d, fpr[sigma_ind], color=c,
                        marker='o', markersize=4, ls='')
            color_list.append(c)

    def exp_fit_func(x, a, b, c):
        return a * np.expt(-b * x) + c

    labels_list = [[] for x in range(configs.nSigmaBins)]
    for sigma_ind, sigma_fpr in enumerate(fpr):
        # pull out the monotonically decreasing part of the fpr curves --
        # currently not used for anything
        monotonic_inds = np.where(np.diff(sigma_fpr) > 0)[0] + 1
        if len(monotonic_inds) > 0:
            d1 = d[:monotonic_inds[0]]
            sigma_fpr1 = sigma_fpr[:monotonic_inds[0]]
            for fpr_ind in range(monotonic_inds[0], len(sigma_fpr)):
                if sigma_fpr[fpr_ind] < sigma_fpr1[-1]:
                    d1 = np.append(d1, d[fpr_ind])
                    sigma_fpr1 = np.append(sigma_fpr1, sigma_fpr[fpr_ind])
        else:
            # all monotonic
            d1 = d
            sigma_fpr1 = sigma_fpr

        sigma_level = np.around(
            configs.ON_THRESHOLD + sigma_ind * (1. / configs.N_BINS_PER_SIGMA), decimals=3)
        labels_list[sigma_ind] = '%s Sigma -- ' % (str(sigma_level))

        fit = None  # curve to plot if interpolation is performed
        for p_ind, p_val in enumerate(p):

            if sigma_fpr[0] < p_val:
                threshold = 0
            elif len(np.where(sigma_fpr < p_val)[0]) == 0:
                threshold = np.nan
            else:
                if fit_type is 'exponential':
                    popt, pcov = curve_fit(exp_fit_func, d, sigma_fpr)
                    fit = exp_fit_func(d_fit_axis, popt[0], popt[1], popt[2])
                    threshold = -1 * np.log(
                        (p_val - popt[2]) / popt[0]) / popt[1]
                    if threshold < 0:
                        threshold = 0

                elif fit_type is 'polynomial':
                    z = np.polyfit(d, sigma_fpr, 4)
                    f = np.poly1d(z)
                    fit = f(d_fit_axis)

                    roots = (f - p_val).r
                    real_positive_roots = roots[
                        np.isreal(roots) * roots.real > 0]
                    if len(real_positive_roots) > 0:
                        threshold = np.amin(real_positive_roots)
                    else:
                        threshold = np.nan

                elif fit_type is 'pw_linear':
                    f = interp1d(sigma_fpr1[::-1], d1[::-1],
                                 kind='linear', bounds_error=False)

                    d_fit_axis = d1
                    fit = sigma_fpr1

                    threshold = f(p_val)

            thresholds[sigma_ind, p_ind] = threshold

            threshold_str = str(
                np.around(thresholds[sigma_ind, p_ind], decimals=3))
            if p_ind == len(p) - 1:
                labels_list[
                    sigma_ind] += 'p=%s: %s' % (str(p_val), threshold_str)
            else:
                labels_list[
                    sigma_ind] += 'p=%s: %s, ' % (str(p_val), threshold_str)

        if return_figs:
            if fit is not None:
                fpr_ax.plot(d_fit_axis, fit, color=color_list[sigma_ind])

    if return_figs:
        fpr_ax.set_ylim((-.05, 0.8))
        fpr_ax.set_xlabel('Duration (sec)')
        fpr_ax.set_ylabel('False positive rate (p)')
        fpr_ax.set_title('False positive rates by sigma and duration')
        plotting.stackedText(fpr_ax, textList=labels_list,
                             colors=color_list, loc=1)

        figs = np.hstack([figs, fpr_fig])
        return thresholds, noise, figs
    else:
        return thresholds, noise

def identify_transients(experimentList, thresholds, noise=None, channel='Ch2',
                        label=None, demixed=False, save=False):

    transients = {}
    for expt in experimentList:
        signals = expt.signals(signal_type='dfof', label=label, channel=channel)
        imData = signals.to_numpy()[...,None]
        
        (nCells, _, nCycles) = imData.shape

        if noise is None:
            exp_noise = estimate_noise(expt, exclude_transients=False)
        else:
            exp_noise = noise[expt.trial_id]

        frame_period = expt.frame_period
        
        expt_transients = []
        
        for cell_idx, roi_label, cell_data, sigma in zip(
                it.count(), signals.index, imData, exp_noise):
            
            cell_transients = [] #SingleCellTransients([], name=roi_label)

            for cycle_idx, cycle_data in zip(it.count(), cell_data.T):
                for start, stop in identify_events(cycle_data / sigma):
                    amp = np.nanmax(cycle_data[start:stop + 1])
                    if np.isnan(amp):
                        continue
                    dur = (stop - start) * frame_period

                    sigma_bin_ind = int(np.floor(configs.N_BINS_PER_SIGMA *
                                        (amp / sigma)) - configs.ON_THRESHOLD *
                                        configs.N_BINS_PER_SIGMA)
                    if sigma_bin_ind > configs.nSigmaBins - 1:
                        sigma_bin_ind = configs.nSigmaBins - 1

                    if dur > thresholds[sigma_bin_ind] and \
                            dur > configs.MIN_DURATION:
                        
                        rel_max_ind = np.where(
                            cycle_data[start:stop + 1] == amp)[0].tolist()[0]
                        transient = Transient(onset=start, offset=stop, sigma=sigma, 
                                          duration=dur, amplitude=amp, 
                                          amplitude_index=start + rel_max_ind)
                        
                        cell_transients.append(transient)

                cell_transients = SingleCellTransients(cell_transients, name=roi_label)
                
            if len(cell_transients) > 0:
                expt_transients.append(cell_transients)
            

        expt_transients = ExperimentTransientGroup(expt_transients, name=expt.trial_id)
            
        if save:
            expt_transients.save(expt.imaging_events_path, event_type='transients', 
                                 label=label, channel=channel)

        transients[expt.trial_id] = expt_transients
        
    return transients

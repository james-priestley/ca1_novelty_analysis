"""Temporary dump of functions from Zhenrui's imaging/behavior data pairing
script, to fold it into the functionality of the new ExperimentGroup objects.

TODO : clean up and remove some of the redundancy. Put the remaining
necessary functions in experiment.utils?
"""

import os
import dateutil
from xml.etree import ElementTree as ET

import pandas as pd
import numpy as np
import scipy.optimize as opt

from lab3 import BehaviorExperiment


def get_xmls(toplevel, mouse_name='', keywords=()):
    xmls = []
    for dirpath, dirnames, filenames in os.walk(toplevel):
        if mouse_name in dirpath:
            for f in filenames:
                keywords_in_f = True
                for kwd in keywords:
                    keywords_in_f *= kwd in f
                if keywords_in_f and '.xml' in f and f[0] != '.':
                    xmls.append(os.path.join(dirpath, f))
    return xmls


def xml2sima(xml):
    sima_path = os.path.splitext(xml)[0] + '.sima'
    if os.path.isdir(sima_path):
        return sima_path
    else:
        return np.nan


def get_xml_timestamp(xml):
    try:
        tree = ET.parse(xml)
    except Exception:
        from lxml import etree
        parser = etree.XMLParser(recover=True)
        tree = etree.parse(xml, parser=parser)
    root = tree.getroot()
    ts = dateutil.parser.parse(root.get('date'))
    return ts


def build_xml_series(toplevel, mouse_name='', keywords=()):
    timestamp_df = pd.DataFrame(get_xmls(toplevel, mouse_name=mouse_name,
                                         keywords=keywords),
                                columns=['xml_path'])
    timestamp_df['sima_path'] = timestamp_df['xml_path'].map(xml2sima)
    timestamp_df['timestamp'] = timestamp_df['xml_path'].map(get_xml_timestamp)
    del timestamp_df['xml_path']
    timestamp_df = timestamp_df.dropna()
    return timestamp_df.set_index('sima_path')


def build_expt_series(mouse):
    """
    Parameters
    ----------
    mouse : lab3.experiment.group.Mouse
    """
    return pd.DataFrame([{'trial_id': expt.trial_id,
                          'timestamp': expt.start_time}
                         for expt in mouse]).set_index('trial_id')


def find_pairs(xml_ts, expt_ts, max_dt=100):
    def abs_time_difference(x):
        return np.abs((xml_ts.loc[x[0]]
                       - expt_ts.loc[x[1]]).timestamp.total_seconds())

    product = pd.MultiIndex.from_product([xml_ts.index, expt_ts.index])
    diffs = pd.Series(product.map(abs_time_difference), index=product)
    diffs = diffs[diffs < max_dt]
    cost_mtx = diffs.unstack()
    cost_mtx = cost_mtx.fillna(100000)
    pairs = opt.linear_sum_assignment(cost_mtx)
    square_df = cost_mtx.iloc[pairs]
    paired_expts = pd.DataFrame(np.diag(square_df),
                                index=[square_df.index, square_df.columns],
                                columns=['tdiff'])
    paired_expts = paired_expts[paired_expts['tdiff'] <= max_dt]
    print(f"Found {len(paired_expts)} pairs from {len(xml_ts)} simas and "
          + f"{len(expt_ts)} behavior experiments")

    unpaired_expts = set(xml_ts.index) \
        - set(paired_expts.index.get_level_values('sima_path'))
    for sima_path in unpaired_expts:
        print(f"FAILED TO PAIR: {sima_path}")

    return paired_expts.reset_index()


def pair_database(pair, xml_ts, expt_ts, verbose=False, do_nothing=False,
                  **kwargs):
    extra_info = ""
    if verbose:
        extra_info += f"(IMAGING: {xml_ts.loc[pair.sima_path].timestamp}"
        extra_info += ", "
        extra_info += f"BEHAVIOR: {expt_ts.loc[pair.trial_id].timestamp})"

    print(f"Pairing Trial: {pair.trial_id} with {pair.sima_path} "
          + f"(DELTA={pair.tdiff} s) {extra_info}")
    if not do_nothing:
        expt = BehaviorExperiment(pair.trial_id)
        expt.pair_imaging_data(pair.sima_path, **kwargs)


def pair_mouse_experiments(mouse, imaging_path, max_dt=60, force_pairing=False,
                           keywords=(), do_nothing=False, verbose=False):
    """
    Parameters
    ----------
    mouse : lab3.experiment.group.Mouse
        Mouse to process
    imaging_path : str
        Locate Prairie xmls/sima folders below this folder
    max_dt : float, optional
        Maximum timestamp difference (in seconds) between behavior tdml and
        Prairie xml files. Defaults to 60
    force_pairing : bool, optional
        Whether to overwrite existing sima pairs. Defaults to False
    keywords : iterable of strs, optional
        Additional keywords to identify in sima_paths
    do_nothing : bool, optional
        Find pairs but don't update the database. Defaults to False
    verbose : bool, optional
        Display extra information. Defaults to False
    """

    print(f"Getting behavior timestamps for {mouse.name}...")
    expt_ts = build_expt_series(mouse)

    print(f"Getting imaging xmls for {mouse.name}...")
    xml_ts = build_xml_series(imaging_path, mouse_name=mouse.name,
                              keywords=keywords)

    print("Pairing using Hungarian algorithm...")
    pairs = find_pairs(xml_ts, expt_ts, max_dt=max_dt)
    pairs.apply(pair_database, axis=1, xml_ts=xml_ts, expt_ts=expt_ts,
                verbose=verbose, do_nothing=do_nothing,
                force_pairing=force_pairing)

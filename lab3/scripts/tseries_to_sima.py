""" Imaging experiment classes require a sima path, rather than a t-series path.
This script allows the user to pass either a mouse or trial ID, and
a sima_path will be added to the attributes of the experiment.

"""
import os
import sys
import argparse

from lab3.experiment.base import \
    BehaviorExperiment, ImagingExperiment, fetch_trials


def get_sima_path_from_t_series(trial_id):
    expt = BehaviorExperiment(trial_id)
    attrib = expt.attrib
    if 'tSeries_path' in attrib.keys():
        t_series_path = attrib['tSeries_path']
        if t_series_path is not None:
            sima_paths = [x for x in os.listdir(t_series_path)
                          if x.endswith('.sima')]
            if len(sima_paths) > 1:
                print('More than one sima path, skipping trial %s' % trial_id)
            elif len(sima_paths) == 0:
                print('No sima folder in the t series, skipping trial %s'
                      % trial_id)
            else:
                sima_path = os.path.join(t_series_path, sima_paths[0])
                return sima_path


def main(argv):
    argParser = argparse.ArgumentParser()
    argParser.add_argument(
        '-m', '--mouse', action='store', type=str, help='mouse name')
    argParser.add_argument(
        '-t', '--trial_id', action='store', type=str, help='trial_id')
    args = argParser.parse_args(argv)

    trial_ids = []
    if args.mouse and args.trial_id:
        print('Pass either a mouse or a trial id, not both')
    elif args.mouse:
        trial_ids = fetch_trials(mouse_name=args.mouse)
    else:
        trial_ids.append(args.trial_id)

    trial_ids_without_sima = []
    for trial_id in trial_ids:
        expt = BehaviorExperiment(trial_id)
        if 'sima_path' not in expt.attrib.keys():
            if 'tSeries_path' in expt.attrib.keys():
                trial_ids_without_sima.append(trial_id)
    for trial_id in trial_ids_without_sima:
        sima_path = get_sima_path_from_t_series(trial_id)
        expt = ImagingExperiment(trial_id, sima_path, store=True)


if __name__ == '__main__':
    main(sys.argv[1:])

import argparse

from lab3 import BehaviorExperiment
from lab3.experiment.group import Mouse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pair behavior experiments with imaging using the "
                    + "Hungarian algorithm")
    parser.add_argument(
        'project_name', type=str, help='sql database name')
    parser.add_argument(
        '--mice', '-m', type=str, nargs='+', help='mice to process')
    parser.add_argument(
        '--directory', '-d', type=str,
        help='directory to search for imaging xmls')
    parser.add_argument(
        '--keywords', '-k', type=str, nargs='+', default=(),
        help='keywords in TSeries name to look for')
    parser.add_argument(
        '--dt', '-t', type=float, default=60,
        help='maximum time difference to pair (seconds)')
    parser.add_argument(
        '--force_pairing', '-f', action='store_true',
        help='flag to overwrite experiments that are already paired')
    parser.add_argument(
        '--dry_run', action='store_true',
        help='print pairs and quit without writing')
    parser.add_argument(
        '--verbose', '-v', action='store_true', help='print extra information')
    args = parser.parse_args()

    for mouse_name in args.mice:

        m = Mouse.from_database(mouse_name, project_name=args.project_name,
                                expt_class=BehaviorExperiment)
        m.pair_imaging_data(
            imaging_path=args.directory,
            max_dt=args.dt,
            force_pairing=args.force_pairing,
            keywords=args.keywords,
            do_nothing=args.dry_run,
            verbose=args.verbose
        )

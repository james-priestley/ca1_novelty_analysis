"""Convenience script for converting raw behavior data from BehaviorMate and
creating associated entries in the experiment database. This script replaces
the formerly separate `tdmlPickler` and `sqlTrialLoader scripts`."""

import argparse
import sys

from lab3 import BehaviorExperiment
from lab3.misc.tdml_pickler import find_files


def main(argv):
    argParser = argparse.ArgumentParser()
    argParser.add_argument(
        '-r', '--repickle', action='store_true',
        help='Delete and recreate the sql entry for this behavior file')
    argParser.add_argument(
        '-o', '--overwrite', action='store_true',
        help='Delete and recreate the database entry for this behavior file')
    argParser.add_argument(
        'filename', action='store', type=str,
        help='Path to tdml file or directory to walk looking for tdml files')
    args = argParser.parse_args(argv)

    # find tdml files, including ones that are already pickled for now
    # (the user may wish to overwrite those database entries still)
    files = find_files(args.filename, include_pickled=True)

    for tdml_path in files:
        try:
            # For each tdml file, (re-)pickle the behavior data and
            # (over)write information to the database
            BehaviorExperiment.create(tdml_path,
                                      repickle=args.repickle,
                                      overwrite=args.overwrite)
        except Exception as e:
            print(f"Error encountered while processing {tdml_path}: \n{e}")


if __name__ == '__main__':
    main(sys.argv[1:])

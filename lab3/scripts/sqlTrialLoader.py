import os.path
import argparse
import sys

from lab3.experiment.database import add_experiment_to_database


def find_files(directory):
    paths = []
    if os.path.splitext(directory)[1] == '.pkl':
        paths.append(directory)
    else:
        for dirpath, dirnames, filenames in os.walk(directory):
            paths.extend(
                map(lambda f: os.path.join(dirpath, f),
                    filter(lambda f: os.path.splitext(f)[1] == '.pkl',
                    filenames)))
    return paths


def main(argv):
    argParser = argparse.ArgumentParser()
    argParser.add_argument(
        '-o', '--overwrite', action='store_true',
        help='delete and recreate the sql entry for this behavior file')
    argParser.add_argument(
        'filename', action='store', type=str,
        help=('path to a pkl file or directory to walk loking for pkl files ' +
              'with trial_info to upload to the experiments database'))
    args = argParser.parse_args(argv)

    files = find_files(args.filename)
    for f in files:
        add_experiment_to_database(f, overwrite=args.overwrite)


if __name__ == '__main__':
    main(sys.argv[1:])

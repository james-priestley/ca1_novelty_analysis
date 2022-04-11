"""A helper script that adds missing h5 metadata for imaging datasets. If the
h5 file does not have a sima wrapper, this is also created."""

import os
import sys
import h5py
import argparse

from lab3.experiment.utils import update_h5_metadata
from sima import Sequence, ImagingDataset


KEY = '/imaging'


def process_h5(h5_path, create_sima_folder=True):
    """Adds 'frame_period' and 'imaging_system' keys to h5 metadata, and
    optionally wraps h5 with a sima ImagingDataset

    Parameters
    ----------
    h5_path : str
        Path to h5 imaging dataset
    create_sima_folder : bool, optional
        Whether to create sima wrapper for dataset, if not already present.
        Defaults to True
    """

    metadata = update_h5_metadata(h5_path)
    print("Updated metadata: %s" % h5_path)

    if not os.path.isdir(h5_path.replace('.h5', '.sima')) \
            and create_sima_folder:

        try:
            channel_names = [i.decode() for i in metadata['channel_names']]
        except AttributeError:
            channel_names = metadata['channel_names']

        seq = Sequence.create("HDF5", h5_path, "tzyxc")
        ds = ImagingDataset([seq], h5_path.replace('.h5', '.sima'),
                            channel_names=channel_names)
        print("    Created SIMA ImagingDataset: %s" % ds.savedir)


def get_h5_paths(directory):
    """Walk through directory and find all h5s that contain imaging data"""

    h5_paths = []
    for dir_name, subdir_list, file_list in os.walk(directory):
        for fname in file_list:
            if fname.endswith('.h5') and '._' not in fname:
                h5_path = os.path.join(dir_name, fname)
                try:  # verify that it's an imaging dataset
                    h5 = h5py.File(h5_path, 'r', libver='latest')
                    h5[KEY]
                    h5_paths.append(h5_path)
                except KeyError:
                    pass
                except OSError:
                    print('Unable to access %s' % h5_path)
                    continue
                h5.close()
    return h5_paths


def main(argv):
    argParser = argparse.ArgumentParser()
    argParser.add_argument("--skip-create", "-s", action="store_true", 
        help="Skip creating sima folders")
    argParser.add_argument(
        "directory", action="store", type=str,
        help=("Directory in which to search for h5 files to update"))
    args = argParser.parse_args(argv)

    h5_paths = get_h5_paths(args.directory)
    for path in h5_paths:
        try:
            process_h5(path, create_sima_folder=not args.skip_create)
        except Exception as exc:
            print(f"Error `{exc}` encountered while processing {path}")
            pass


if __name__ == '__main__':
    main(sys.argv[1:])

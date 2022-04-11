import os
import shutil
import datetime


def backup_dir(directory, delete_original=False):
    directory = directory.rstrip(os.sep)
    backup = '{}.backup_{}'.format(
        directory, datetime.datetime.today().strftime('%Y-%m-%dT%H-%M-%S'))

    if delete_original:
        shutil.move(directory, backup)
    else:
        shutil.copytree(directory, backup)


def backup_sima(imds):
    sima_dir = imds.savedir
    backup_dir(sima_dir)


def get_base_sequence(seq):
    """Strips away any existing sequence wrappers to access the raw data"""
    base = seq
    while True:
        try:
            base = base._base
        except AttributeError:
            return base

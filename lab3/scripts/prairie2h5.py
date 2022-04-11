import os
import sys
from glob import glob
import time
import argparse
import shutil

import numpy as np
import xml.etree.ElementTree as ET
#import bottleneck as bn
import h5py


GROUP = '/'
KEY = 'imaging'
NAME = os.path.join(GROUP, KEY)
NBYTES = 2
DTYPE = np.int16


def get_frame_info(xml_root):
    """Returns frame dimension and sampling information"""
    for f in xml_root.find("PVStateShard").findall("PVStateValue"):
        if f.attrib['key'] == "linesPerFrame":
            ypx = int(f.attrib['value'])
        elif f.attrib['key'] == "pixelsPerLine":
            xpx = int(f.attrib['value'])
        elif f.attrib['key'] == "micronsPerPixel":
            for child in f:
                if child.attrib['index'] == 'XAxis':
                    xsize = float(child.attrib['value'])
                if child.attrib['index'] == 'YAxis':
                    ysize = float(child.attrib['value'])
        elif f.attrib['key'] == "resonantSamplesPerPixel":
            samplesPerPixel = int(f.attrib['value'])

    return ypx, xpx, xsize, ysize, samplesPerPixel


def get_recording_info(sequences):
    "Returns frame/plane/channel counts/names"
    xml_frames = [seq.findall('Frame') for seq in sequences]
    if len(xml_frames) == 1:
        n_planes = 1
        n_frames = len(xml_frames[0])
    else:
        n_planes = len(xml_frames[0])
        n_frames = len(xml_frames)
    n_channels = len(xml_frames[0][0].findall('File'))
    channel_names = [f.attrib['channelName']
                     for f in xml_frames[0][0].findall('File')]

    return n_frames, n_planes, n_channels, channel_names


def prepare_h5(xml_name, output_shape, channel_names, ysize, xsize):
    """Creates the h5 file"""
    h5 = h5py.File(xml_name.replace('.xml', '.h5'), 'w', libver='latest')
    h5[GROUP].create_dataset(
        KEY, output_shape, DTYPE, maxshape=output_shape,
        chunks=(1, 1, output_shape[2], output_shape[3], 1), compression=None)
    for idx, label in enumerate(['t', 'z', 'y', 'x', 'c']):
        h5[NAME].dims[idx].label = label
    h5[NAME].attrs['channel_names'] = [ch.encode("utf8") for ch in channel_names]
    h5[NAME].attrs['element_size_um'] = np.array([1, ysize, xsize])

    return h5


def find_raw_files(xml_name):
    """Finds prairie raw files associated with the xml"""
    return sorted(glob(os.path.join(os.path.split(xml_name)[0],
                                    "CYCLE_000001_RAWDATA_" + '*')))


def process_binaries(xml_name, h5, samplesPerPixel):
    """Reads binary files, formats data, and saves to h5.
    Data formatting is adapted from haussio.py, from:
     https://github.com/neurodroid/haussmeister
    """

    n_frames, n_planes, ypx, xpx, n_channels = h5[KEY].shape
    frame_size = xpx * ypx * NBYTES * samplesPerPixel * n_channels * n_planes
    rem_data = b''
    current_frame = 0
    raw_files = find_raw_files(xml_name)
    checksum = 0

    print('   Reading raw files...')
    for ridx, rf in enumerate(raw_files):

        with open(rf, 'rb') as rawf:
            print('      ({}/{}) '.format(ridx+1, len(raw_files)) + rf)
            raw_data = rem_data + rawf.read()

        stray_bytes = len(raw_data) % (frame_size)

        if stray_bytes > 0:
            current_data = np.frombuffer(
                raw_data[:-stray_bytes], dtype=DTYPE).reshape(
                    len(raw_data[:-stray_bytes]) // (frame_size // n_planes),
                    ypx, xpx, samplesPerPixel, n_channels)
            rem_data = raw_data[-stray_bytes:]
        else:
            current_data = np.frombuffer(raw_data, dtype=DTYPE).reshape(
                len(raw_data) // (frame_size // n_planes),
                ypx, xpx, samplesPerPixel, n_channels)
            rem_data = b''

        # format data
        current_data = current_data.astype(np.float) - 2**13  # 13 bit offset
        current_data[current_data < 0] = np.nan  # remove negative values

        # average multisamples, remove nans
        #current_data = bn.nanmean(current_data, axis=-2)
        current_data = np.nanmean(current_data, axis=-2)
        current_data[np.isnan(current_data)] = 0

        # resonant correction and tzyxc reshaping
        current_data[:, 1::2, ::] = np.flip(current_data[:, 1::2, ::], axis=2)
        current_data = current_data.reshape(-1, n_planes, ypx, xpx, n_channels)
        current_data = current_data.astype(DTYPE)

        # save to h5
        h5[KEY][current_frame:(current_frame+current_data.shape[0])] = \
            current_data[0:(n_frames - current_frame)]

        checksum += \
            np.sum(current_data[0:(n_frames - current_frame), :, 0, 0, :]) + \
            np.sum(current_data[0:(n_frames - current_frame), :, -1, -1, :])
        current_frame += current_data.shape[0]

    h5sum = np.sum(h5[KEY][:, :, 0, 0, :]) + np.sum(h5[KEY][:, :, -1, -1, :])

    return h5sum, checksum, raw_files


def convert_to_HDF5(directory, xml_name, delete=False, move_dir=None):
    """Function to convert Prairie binary files to HDF5

    Parameters
    ----------
    directory : str
    xml_name : str
        Path to Prairie XML.
    delete : bool
        Whether to delete the Prairie binary files if the conversion is
        successful.
    move_dir : str
        If not None, after successful completion of conversions, move the
        parent directory of the h5 files to 'move_dir', mirroring the relative
        path from 'directory'.
    """

    start_time = time.time()

    print('Converting: {}'.format(xml_name))

    sequences = (elem for _, elem in ET.iterparse(xml_name)
                 if elem.tag == 'Sequence' and elem.get('type') !=
                 'TSeries Voltage Output Experiment')

    xml_root = ET.parse(xml_name).getroot()

    # get frame and recording information
    ypx, xpx, xsize, ysize, samplesPerPixel = get_frame_info(xml_root)
    n_frames, n_planes, n_channels, channel_names = \
        get_recording_info(sequences)

    # convert to h5
    print(f"   Frames: {n_frames}\n   Planes: {n_planes}\n"
          + f"   Y-pixels: {ypx}\n   X-pixels: {xpx}\n"
          + f"   Channels: {n_channels}")

    print(f"   Channel names: {channel_names}")
    h5 = prepare_h5(xml_name, (n_frames, n_planes, ypx, xpx, n_channels),
                    channel_names, ysize, xsize)
    h5sum, checksum, raw_files = process_binaries(xml_name, h5,
                                                  samplesPerPixel)

    # verify the checksum
    if h5sum != checksum:
        h5.close()
        os.remove(xml_name.replace('.xml', '.h5'))
        if h5sum > -1:  # if -1, there were no binaries to convert
            print(f"   FAILED checksum, got {h5sum} in h5, "
                  + f"expected {checksum}")

            # TODO put a fail file in the directory

    else:
        print(f"   Passed checksum, successfully created {h5}")
        h5.close()

        if delete:
            print("   Deleting binary files...")
            for f in raw_files:
                os.remove(f)

        if move_dir:
            cur_dir = os.path.split(xml_name)[0]
            rel_path = os.path.relpath(cur_dir, directory)
            new_path = os.path.join(move_dir, rel_path)
            if os.path.isdir(new_path):
                print(f"FAILED moving {cur_dir} to {new_path}: "
                      + "path already exists")
            else:
                try:
                    shutil.move(cur_dir, new_path)
                except Exception:
                    print(f"FAILED moving {cur_dir} to {new_path}")
                else:
                    print(f"Successfully moved {cur_dir} to {new_path}")

    elapsed = time.time() - start_time
    print(f"Completed in {elapsed} seconds\n")


def main(argv):

    argParser = argparse.ArgumentParser()
    argParser.add_argument(
        "-d", "--delete", action="store_true",
        help="Delete converted binaries")
    argParser.add_argument(
        "-o", "--overwrite", action="store_true",
        help="Re-convert binaries with pre-existing h5 files")
    argParser.add_argument(
        "-m", "--move_dir", action="store", type=str, default=None,
        help="When complete move entire contents of parent directory to new "
             + "path. The relative path from 'directory' will be mirrored to "
             + "'move_dir'.")
    argParser.add_argument(
        "directory", action="store", type=str, default=os.curdir,
        help="Locate all Prairie XMLs below this folder and convert files.")

    args = argParser.parse_args(argv)

    # find Prairie xmls
    xml_paths = []
    for dirName, subdirList, fileList in os.walk(args.directory):
        for fname in fileList:
            if 'xml' in fname and '._' not in fname:
                xml_paths.append(os.path.join(dirName, fname))

    # remove XMLs that have an h5 file already
    if not args.overwrite:
        xml_paths = [x for x in xml_paths
                     if not os.path.isfile(x.replace('.xml', '.h5'))]

    # remove XMLs that have no raw files
    xml_paths = [x for x in xml_paths if len(find_raw_files(x)) > 0]

    for pidx, path in enumerate(xml_paths):
        print("{}/{} XMLs".format(pidx+1, len(xml_paths)))
        convert_to_HDF5(args.directory,
                        path,
                        delete=args.delete,
                        move_dir=args.move_dir)


if __name__ == '__main__':
    main(sys.argv[1:])

# todo: paths of files to archive are stored as relative paths from the archive_files.txt file.
# these should be absolute paths to minimize the chance of errors

"""Helper functions to locate and archive old data."""
import argparse
import os
import re
import time
import sys
import numpy as np
from glob import glob
from archive_helper import *

S3_BUCKET = 'losonczylab.data.archive'
# The number of a days a restored file will remain before being deleted. Files restored from Glacier storage are kept in 
# both Glacier and a copy in Reduced Redundancy storage, so for this many days we pay double for the data storage.
DEFAULT_PATTERN = re.compile(".*")
GLACIER_RETENTION_PERIOD = 7
LONG_AGO_EXTRACTED_THRESHOLD = 150  # in days
LARGE_FILE_THRESHOLD = 0.25 # in GB
ARCHIVE_FILE_NAME = 'archive_file.txt'
ARCHIVE = '.archive'
H5 = '.h5'

def getSizeInGB(file_size):
    return round((file_size / 1024. ** 3), 2)

def dangling_h5(directory, fNPattern=DEFAULT_PATTERN):
    """Find h5 files with no corresponding sima directory."""
    total = 0.
    dangling_files = []
    for d, folders, files in os.walk(directory):
        if any(fil.endswith(H5) for fil in files) and not any(fold.endswith('.sima') for fold in folders):
            for f in files:
                if f.endswith(H5):
                    fName = os.path.join(d, f)
                    file_size = getSizeInGB(os.path.getsize(fName))
                    if fNPattern.match(fName):
                        dangling_files.append(fName)
                        print_and_log(f"Dangling h5: {fName}", 'dangling_h5')
                        total += file_size
    print_and_log(f"\nTotal size of dangling HDF5 files: {total} GB\n", 'dangling_h5')
    return dangling_files, total


def all_h5(directory, fNPattern=DEFAULT_PATTERN):
    """Find all HDF5 files."""
    total = 0.
    h5_files = []
    for d, folders, files in os.walk(directory):
        for f in files:
            if f.endswith(H5):
                fName = os.path.join(d, f)
                file_size = getSizeInGB(os.path.getsize(fName))
                if fNPattern.match(fName):
                    h5_files.append(fName)
                    print_and_log(f"HDF5: {fName}", 'all_h5')
                    total += file_size

    print_and_log(f"\nTotal size of all HDF5 files: {total} GB\n", 'all_h5')
    return h5_files, total


def large_files(directory, threshold=LARGE_FILE_THRESHOLD, fNPattern=DEFAULT_PATTERN):
    """Find all large files. Threshold is in GB."""
    total = 0.
    large_files = []
    for d, folders, files in os.walk(directory):
        for f in files:
            fName = os.path.join(d, f)
            file_size = getSizeInGB(os.path.getsize(fName))
            if file_size > threshold:
                if fNPattern.match(fName) and not fName.endswith(f'/{ARCHIVE_FILE_NAME}'):
                    large_files.append(fName)
                    print_and_log(f"Large file: {fName}", 'large_files')
                    total += file_size
    print_and_log(f"\nTotal size of all large files: {total} GB\n", 'large_files')
    return large_files, total


def all_lif(directory, fNPattern=DEFAULT_PATTERN):
    """Find all LIF files."""
    total = 0.
    lif_files = []
    for d, folders, files in os.walk(directory):
        for f in files:
            if f.endswith('.lif'):
                fName = os.path.join(d, f)
                file_size = getSizeInGB(os.path.getsize(fName))
                if fNPattern.match(fName):
                    lif_files.append(fName)
                    print_and_log(f"LIF: {fName}", 'all_lif')
                    total += file_size
    print_and_log(f"\nTotal size of all LIF files: {total} GB\n", 'all_lif')
    return lif_files, total


def long_ago_extracted_h5(directory, threshold=LONG_AGO_EXTRACTED_THRESHOLD, fNPattern=DEFAULT_PATTERN):
    """Find all HDF5 files with signals extracted a long time ago.

    'A long time ago' is defined by the threshold parameter (in days).

    """
    total = 0.
    old_files = []
    for d, folders, files in os.walk(directory):
        if any(fil.endswith(H5) for fil in files):
            sima_dirs = [folder for folder in folders if folder.endswith('sima')]
            h5_files = [fil for fil in files if fil.endswith('h5')]
            if len(sima_dirs) != 1 or len(h5_files) != 1:
                continue
            signals = glob(os.path.join(d, sima_dirs[0], 'signals_?.pkl'))
            if not len(signals):
                continue

            newest_signals = np.inf
            for sig_file in signals:
                age_in_days = (time.time() - os.stat(sig_file).st_mtime) / 60. / 60. / 24.
                newest_signals = np.min([newest_signals, age_in_days])

            if newest_signals > threshold:
                fName = os.path.abspath(os.path.join(d, h5_files[0]))
                file_size = getSizeInGB(os.path.getsize(fName))
                if fNPattern.match(fName):
                    old_files.append(fName)
                    print_and_log(f"Long ago extracted HDF5: {fName}", 'long_ago_extracted_h5')
                    total += file_size
    print_and_log(f"\nTotal size of long ago extracted HDF5 files: {total} GB\n", 'long_ago_extracted_h5')
    return old_files, total


def locate_archived_files(directory, fNPattern=DEFAULT_PATTERN, top_level_only=False):
    """Return a list of all archived files."""
    result = []
    if top_level_only:
        for f in os.listdir(directory):
            if f.endswith(ARCHIVE):
                fName = os.path.abspath(os.path.join(directory, f))
                if fNPattern.match(fName):
                    result.append(fName)
    else:
        for d, folders, files in os.walk(directory):
            for f in files:
                if f.endswith(ARCHIVE):
                    fName = os.path.abspath(os.path.join(d, f))
                    if fNPattern.match(fName):
                        result.append(fName)
    return result


def list_archived_files(directory, fNPattern=DEFAULT_PATTERN):
    """List all archived files."""
    total = 0
    count = 0
    for f in locate_archived_files(directory, fNPattern=fNPattern):
        archived_file_data = parse_placeholder(f)
        file_size = getSizeInGB(archived_file_data.get('stat', {}).get('size'))
        total += file_size
        count += 1
    print_and_log(f"\nTotal size of {count} archived files: {total} GB\n", 'list_archived_files')
    
def aws_installed_and_configured():
    # check awscli has been installed
    try:
        result = subprocess.run(f'which aws', shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        output = result.stdout.decode('utf-8')
        error = result.stderr.decode('utf-8')
        if output == '' or error != '':
            raise RuntimeError()
    except:
        print('AWS CLI is not installed. Please contact the system administrator.\n')
        return False
    # check awscli has been configured
    try:
        result = subprocess.run(f'aws s3 ls s3://{S3_BUCKET}', shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        output = result.stdout.decode('utf-8')
        error = result.stderr.decode('utf-8')
        if output == '' or error != '':
            raise RuntimeError()
    except:
        print('AWS CLI has not been properly configured. Please contact the system administrator.\n')
        return False
    return True

if __name__ == '__main__':
    # check if the AWS CLI has been installed and configured
    if not aws_installed_and_configured():
        exit()
    
    argParser = argparse.ArgumentParser()
    argParser.add_argument(
        'directory', action='store', type=str, nargs='?', default=os.curdir, 
        help="directory to search for or restore datasets. Defaults to the current working directory, i.e. the \
        directory that you are in when launching the script not the directory the script is in.")
    group = argParser.add_mutually_exclusive_group() # only one of the arguments added to this group may be used at a time
    group.add_argument(
        '-s', '--save', action="store_true", help="Save found files to a text file.")
    group.add_argument(
        '-L', '--load', action="store_true", help="Upload all files found in archive.txt to AWS.")
    group.add_argument(
        '-R', '--restore', action="store_true", help="Restore placeholder files in directory.")
    group.add_argument(
        '--list', action="store_true", help="Locate all already archived files.")
    argParser.add_argument(
        '-o', '--overwrite', action="store_true", help="Overwrite the archive_files.txt instead of appending to it.")
    argParser.add_argument(
        '--large', action="store", type=float, default=-1,
        help="Locate files with size greater than this argument. NOTE: passing 0 for this argument will archive all files. \
        Negative values will be ignored.")
    argParser.add_argument(
        '-f', '--archivefile', action="store", type=str, default=os.curdir,
        help="The path to search for the archive_files.txt. When passed with -s or --save, the script will append to or \
        create this file. When passed with -L or --load, the script will read from this file and throw an error if it \
        does not exist. Defaults to the current directory. ")
    argParser.add_argument(
        '-A', '--all_h5', action="store_true", help="Locate all HDF5 files.")
    argParser.add_argument(
        '-T', '--toplevelonly', action="store_true", 
        help="Don't restore archive files in subdirectories. Only valid when used with -R/--restore.")
    argParser.add_argument(
        "--regexp", action="store", type=str,
        help="specify a python regular expression pattern, only files (the absolute path of the file) that matches the \
        pattern are selected. Ex: \".*\.h5\" would find all files that ends with \".h5\". See python re for the regular \
        expression documentation.")
    args = argParser.parse_args()
    
    if '..' in args.directory:
        print('Directory argument cannot contain \'..\'\n')
        exit()
        
    if args.load and args.regexp:
        print('--regexp argument cannot be used with -L/--load argument.\n')
        exit()
        
    if args.toplevelonly and not args.restore:
        print('--toplevelonly argument can only be used with -R/--restore argument.\n')
        exit()
        
    # convert relative directory path to absolute path for args.directory and args.archivefile
    if args.directory.startswith('.'):
        args.directory = os.path.abspath(args.directory)
    
    archive_file = os.path.join(args.archivefile, ARCHIVE_FILE_NAME)
    if archive_file.startswith('.'):
        archive_file = os.path.abspath(archive_file)
    
    print_and_log(f'Running archive helper script on directory {args.directory}', 'main')
    
    if args.regexp:
        try:
            fNPattern = re.compile(args.regexp) # may need to do some error checking with this
        except:
            print_and_log(f'Invalid regular expression: {args.regexp}\n', 'main')
            exit()
    else:
        fNPattern = DEFAULT_PATTERN

    # list previously archived files
    if args.list:
        list_archived_files(args.directory, fNPattern=fNPattern)
        
    # restore archived files
    elif args.restore:
        files_to_restore = locate_archived_files(args.directory, fNPattern=fNPattern, top_level_only=args.toplevelonly)
        restore(files_to_restore) 
        
    # Write the full paths of all files to be archived to archive_files.txt
    elif args.save:
        files = []
        
        # Dangling HDF5 files
        dangling_files, totalSize = dangling_h5(args.directory, fNPattern=fNPattern)
        if len(dangling_files):
            files.extend(['#', f'# Dangling HDF5 files in {args.directory}', f'# {totalSize}GB', "#"])
            files.extend(sorted(dangling_files))
            
        # Leica confocal images
        lif_files, totalSize = all_lif(args.directory, fNPattern=fNPattern)
        if len(lif_files):
            files.extend(['#', f'# LIF files in {args.directory}', f'# {totalSize}GB', "#"])
            files.extend(sorted(lif_files))
  
        # HDF5 files extracted long ago
        old_files, totalSize = long_ago_extracted_h5(args.directory, threshold=LONG_AGO_EXTRACTED_THRESHOLD, fNPattern=fNPattern)
        if len(old_files):
            files.extend(['#', f'# Long ago extracted HDF5 files in {args.directory}', f'# {totalSize}GB', "#"])
            files.extend(sorted(old_files))
 
        # Large files
        if args.large >= 0:
            large_files, totalSize = large_files(args.directory, threshold=args.large, fNPattern=fNPattern)
            if len(large_files):
                files.extend(['#', f'# Large files in {args.directory}', f'# {totalSize}GB', "#"])
                files.extend(sorted(large_files))
 
        # All HDF5 files
        if args.all_h5:
            all_h5, totalSize = all_h5(args.directory, fNPattern=fNPattern)
            if len(all_h5):
                files.extend(['#', f'# All HDF5 files in {args.directory}',f'# {totalSize}GB', "#"])
                files.extend(sorted(all_h5))
            
        if args.overwrite:
            mode = 'w'
        else:
            mode = 'a'
        
        with open(archive_file, mode) as f:
            for file in files:
                # convert relative paths to absolute paths, then write to file
                if not file.startswith('#'):
                    file = os.path.abspath(file)
                f.write(f"{file}\n")
     
    # upload local files to AWS and replace them with placeholder files
    elif args.load:
        files_to_archive = set() # use a set to prevent duplicates
        if not os.path.isfile(archive_file):
            print_and_log(f"File does not exist: {archive_file}. Exiting script.\n", 'main')
            exit()
        with open(archive_file, 'r') as f:
            for line in f:
                line_stripped = line.strip()
                if not line_stripped or line_stripped.startswith('#'):
                    continue
                files_to_archive.add(line_stripped)
        archive(sorted(files_to_archive))
        
    else:
        print_and_log("\nMissing required parameters. Must provide exactly one of the following: " \
        "-s/--save, -L/--load, -R/--restore, --list.\n", 'main')

    close_log_file()
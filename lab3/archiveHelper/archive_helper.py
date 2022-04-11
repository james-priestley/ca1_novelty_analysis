"""Losonczy Lab data archive utilities.

These utilities can be used to archive and restore files to a remote host.

To get started, you need to setup an account with AWS IAM that grants you access to the lab S3 bucket and load your 
authentication keys. To do this, go to http://129.236.191.58:12000/wiki/AWS and refer to the "Installing and Configuring
AWS CLI" section.

To archive a file:
>>> archive(['/data/user/mouse1/TSeries1/TSeries1.h5'])

To restore a file:
>>> restore(['/data/user/mouse1/TSeries1/TSeries1.h5.archive'])

"""
import os
import json
import subprocess
import time
from datetime import datetime
from json import JSONDecodeError

S3_BUCKET = 'losonczylab.data.archive'

# The number of a days a restored file will remain before being deleted. Files restored from Glacier or Deep Archive 
# storage are kept in both Glacier or Deep Archive and Reduced Redundancy storage, so for this many days we pay double
# for storage.
RESTORE_DURATION = 7
ONE_HOUR = 3600
TWENTY_FOUR_HOURS = 86400
LOG_FILE_NAME = f'archive_helper.log'
LOG_FILE = None

def archive(files_to_archive, bucket=S3_BUCKET, storage_class='DEEP_ARCHIVE'):
    """Upload files to an AWS S3 bucket and write a placeholder file for each uploaded file.

    Parameters
    ----------
    files_to_archive : list
        List of paths to the files to be archived, will be mirrored in S3.
    bucket : str, optional
        Name of S3 bucket to upload files to.
    storage_class : {'STANDARD', 'GLACIER', 'DEEP_ARCHIVE'}, optional
        Initial storage class of files. Lifecycle rules on the bucket might change this.
    """
    for local_path in files_to_archive:
        local_path = os.path.abspath(local_path)
        remote_path = local_path.lstrip('/')
        archive_successful = archive_file(remote_path, local_path, bucket, storage_class)
        if archive_successful:
            write_placeholder(local_path, remote_path, bucket)
            if os.path.exists(local_path):
                os.remove(local_path)
            else:
                print_and_log(f'File not found: {local_path}\n', 'archive')

            
def archive_file(remote_path, local_path, bucket, storage_class):
    """Upload a file to an AWS S3 bucket.
    
    Parameters
    ----------
    remote_path: str
        Remote path of file.
    local_path: str
        Local path of file.
    bucket: str
        Name of S3 bucket to upload file to.
    storage_class: {'STANDARD', 'GLACIER', 'DEEP_ARCHIVE'}
        Initial storage class of file. Lifecycle rules on the bucket might
        change this.
    """
    result = subprocess.run(f'aws s3 cp {local_path} s3://{bucket}/{remote_path} --storage-class {storage_class}',
                            shell=True, stdout=subprocess.PIPE).stderr
    if result is not None:
        return False
    print_and_log(f'Archive successful: {remote_path}\n', 'archive_file')
    return True
    
def restore(files_to_restore, bucket=S3_BUCKET, restore_duration=RESTORE_DURATION):
    """Restore files from an AWS S3 bucket.

    Parameters
    ----------
    files_to_restore : list
        List of paths to placeholder files (.archive) that contain information on the remote location of the file to restore.
    restore_in_place : bool, optional
        If True, ignore the stored local file path in the placeholder file and restore the file in the same location as the 
        placeholder file.
    bucket : str, optional
        Name of S3 bucket to upload files to.
    restore_duration: int, optional
        The number of a days restored files will remain available for
        downloaded.
    """
    restore_paths = []
    for placeholder_file in files_to_restore:
        data = parse_placeholder(placeholder_file)
        restore_path = {'remote_path': data['remote_path'], 'restore_complete': False}
        restore_path['local_path'] = data['local_path']
        restore_paths.append(restore_path)
    total_wait = 0
    while True:
        all_files_restored = True
        for entry in restore_paths:
            restore_complete = entry['restore_complete']
            local_path = entry['local_path']
            remote_path = entry['remote_path']
            if not restore_complete:
                if restore_file(remote_path, local_path, bucket, restore_duration):
                    entry['remote_path'] = True
                    file_to_remove = f'{local_path}.archive'
                    if os.path.exists(file_to_remove):
                        os.remove(file_to_remove)
                        print_and_log(f'Removed file: {file_to_remove}\n', 'restore')
                    else:
                        print_and_log(f'File not found: {file_to_remove}\n', 'restore')
                else:
                    all_files_restored = False
        if all_files_restored:
            break
        print_and_log('1 or more files has not yet been restored from Glacier or Deep Archive. Waiting one hour...', 'restore')
        time.sleep(ONE_HOUR)
        total_wait += ONE_HOUR
        if total_wait > TWENTY_FOUR_HOURS:
            raise TimeoutError
    print_and_log(f'Restored {len(restore_paths)} files in {total_wait/3600} hours', 'restore')
    
    
def restore_file(remote_path, local_path, bucket, restore_duration):
    """Upload a file to S3.
    
    Parameters
    ----------
    remote_path: str
        Remote path of file.
    local_path: str
        Local path of file.
    bucket: str
        Name of S3 bucket to upload file to.
    storage_class: {'STANDARD', 'GLACIER', 'DEEP_ARCHIVE'}
        Initial storage class of file. Lifecycle rules on the bucket mightchange this.
    """
    query = query_file(remote_path, bucket)
    if query is None:
        return False
    storage_class = query['StorageClass']
    restore_started = glacier_restore_started(query)
    
    if storage_class in ['GLACIER', 'DEEP_ARCHIVE']:
        if not is_restored_from_glacier(query):
            if not glacier_restore_started(query):
                print_and_log(f'Initiating restoring from Glacier: {remote_path}', 'restore_file')
                begin_glacier_restore(remote_path, bucket, restore_duration)
            return False
        
    print_and_log(f'Downloading: {remote_path}\n', 'restore_file')
    restore_successful = begin_standard_restore(remote_path, local_path, bucket=S3_BUCKET)
    if restore_successful:
        print_and_log(f'Restore Successful: {remote_path}\n', 'restore_file')
        return True
    return False


def begin_glacier_restore(remote_path, bucket, restore_duration):
    """Make an AWS CLI call to begin the restore process on a remote file stored in the bucket.
    
    Parameters
    ----------
    remote_path: str
        Remote path to file.
    bucket: str
        Bucket file is stored in.
    restore_duration: str
        Amount of time file will be available for download.
    """
    restore_request = f'\'{{"Days":{restore_duration}, "GlacierJobParameters":{{"Tier":"Standard"}}}}\''
    result = subprocess.run(
        f'aws s3api restore-object --bucket {bucket} --key {remote_path} --restore-request {restore_request}', shell=True)


def query_file(remote_path, bucket):
    """Get metadata on a remote file stored in the bucket.
    
    Parameters
    ----------
    remote_path: str
        Remote path of file.
    bucket: str
        Bucket the file is stored in.
    """
    result = subprocess.run(f'aws s3api head-object --bucket {bucket} --key {remote_path}', shell=True, stdout=subprocess.PIPE)
    try:
        result = json.loads(result.stdout.decode('utf-8'))
    except JSONDecodeError:
        result = None
    return result


def glacier_restore_started(query):
    """Parses a query made on a remote file and returns True if a restore from Glacier or Deep Archive has been initiated.
    
    Parameters
    ----------
    query: str
        String retured from the query_file() function.
    """
    try:
        query['Restore']
        return True
    except KeyError:
        return False


# returns True if the file has been restored from glacier,
# false otherwise
def is_restored_from_glacier(query):
    """Parses a query made on a remote file and returns True if a restore from Glacier or Deep Archive has been completed.
    
    Parameters
    ----------
    query: str
        String retured from the query_file() function.
    """
    try:
        return (query['Restore'] != 'ongoing-request="true"')
    except KeyError:
        return False


def begin_standard_restore(remote_path, local_path, bucket):
    """Make an AWS CLI call to download the remote file stored in the given bucket to the local path specified.
    
    Parameters
    ----------
    remote_path: str
        Remote path of file.
    local_path: str
        Local path to download file to.
    bucket: str
        Bucket remote file is stored in.
    """
    result = subprocess.run(
        f'aws s3 cp s3://{bucket}/{remote_path} {local_path} --force-glacier-transfer', shell=True, stdout=subprocess.PIPE).stderr
    if result is not None:
        return False
    return True

def write_placeholder(local_path, remote_path, bucket):
    """Write placeholder file that references remote location.

    The placeholder file should contain all the information to locate the remote file and also verify that a file 
    downloaded from the remote location matches the original file.

    Parameters
    ----------
    local_path : str
        Local path to file.
    remote_path : str
        Remote path to file.
    """
    placeholder_path = local_path + '.archive'
    if os.path.exists(placeholder_path):
        raise ValueError(f'File already exists: {placeholder_path}')

    st_mode, st_ino, st_dev, st_nlink, st_uid, st_gid, st_size, st_atime, st_mtime, st_ctime = os.stat(local_path)

    data = {
        'stat': {
            'mode': st_mode,
            'ino': st_ino,
            'dev': st_dev,
            'nlink': st_nlink,
            'uid': st_uid,
            'gid': st_gid,
            'size': st_size,
            'atime': convert_time(st_atime),
            'mtime': convert_time(st_mtime),
            'ctime': convert_time(st_ctime)},
        'timestamp': convert_time(time.time()),
        'local_path': local_path,
        'remote_path': remote_path,
        'bucket': bucket
    }
    json.dump(data, open(placeholder_path, 'w'), sort_keys=True, indent=4,
              separators=(',', ': '))

def parse_placeholder(placeholder_path):
    """Returned the parsed contents of a placeholder file."""
    return json.load(open(placeholder_path, 'r'))


def convert_time(time_in_seconds):
    return time.strftime(
        '%Y-%m-%d-%Hh%Mm%Ss', time.localtime(time_in_seconds))

def print_and_log(print_msg, method_name):
    global LOG_FILE
    if LOG_FILE is None:
        try:
            LOG_FILE = open(LOG_FILE_NAME, 'a')
            LOG_FILE.write('--------------------------------------------------------------------------------\n')
            LOG_FILE.write(f'{datetime.now().strftime("%D  %H:%M:%S")}\n\n\n')
        except Exception as e:
            print(Exception, e)
    else:
        try:
            LOG_FILE.write(f'{method_name}(): {print_msg}')
        except Exception as e:
            print(Exception, e)
    print(print_msg)
    
def close_log_file():
    LOG_FILE.close()

from lab3.experiment.base import *
from lab3.experiment.database import ExperimentDatabase
import numpy
import os
import pickle

def compare_dicts(d1, d2):
    for key in d1.keys():
        value = d1[key]
        try:
            value2 = d2[key]
        except KeyError:
            return False
        if type(value) != type(value2):
            return False
        # compare numpy.ndarrays
        if type(value) == numpy.ndarray:
            if (value.shape != value2.shape) or (value.dtype != value2.dtype):
                return False
            elif value.size > 0 and (value != value2).all():
                return False
        # compare dicts
        elif type(value) == dict:
            if not compare_dicts(value, value2):
                return False
        # compare anything else
        elif value != value2:
            return False
    # all key-value pairs compared and found to be identical. Returning true
    return True

def pickle_BehaviorExperiment():
    file_name = 'behaviorExperiments.pickle'
    if os.path.exists(file_name):
        os.remove(file_name)
    experiments = [BehaviorExperiment(12707), BehaviorExperiment(16716)]
    with open(file_name, 'wb') as f:
        pickle.dump(experiments, f)
        
def pickle_ImageExperiment():
    file_name = 'imageExperiments.pickle'
    if os.path.exists(file_name):
        os.remove(file_name)
    trial_id = 999999
    sima_path = '/data3/jack/2p/jb130/20191124/jb130_20191124_0f193278bcf8ac5_1-001/jb130_20191124_0f193278bcf8ac5_1-001.2564997a.sima'
    experiments = [ImagingExperiment(trial_id, sima_path=sima_path)]
    with open(file_name, 'wb') as f:
        pickle.dump(experiments, f)
        
def pickle_all():
    pickle_BehaviorExperiment()
    pickle_ImageExperiment()

def get_table_columns(table):
    db = ExperimentDatabase()
    result = [
        entry['COLUMN_NAME'] for entry in 
        db.select_all(f"SELECT * FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_NAME = N'{table}'")
    ]
    db.disconnect()
    return result
    
def delete_row_from_mice(mouse_id, mouse_name):
    db = ExperimentDatabase()
    max_delete = 1
    num_rows_to_delete = len(db.select_all(f'SELECT * FROM mice WHERE mouse_id = \
                                       "{mouse_id}" and mouse_name = "{mouse_name}"'))
    if num_rows_to_delete > max_delete:
        print(f'Attempting to delete more than {max_delete} rows from table: mice. Exiting without deleting')
    else:
        if db.query(f'DELETE from mice WHERE mouse_id = "{mouse_id}" and mouse_name = "{mouse_name}"'):
            print(f'Deleted {num_rows_to_delete} from mice')
    db.disconnect()
        
def delete_row_from_trials(trial_id, mouse_id, start_time, stop_time, behavior_file, 
                           tSeries_path, experiment_group, experiment_id):
    db = ExperimentDatabase()
    max_delete = 1
    num_rows_to_delete = len(db.select_all(f'SELECT * FROM trials WHERE trial_id = "{trial_id}" and \
                                             mouse_id = "{mouse_id}" and start_time = "{start_time}" and \
                                             stop_time = "{stop_time}" and behavior_file = "{behavior_file}" and \
                                             tSeries_path = "{tSeries_path}" and experiment_group = "{experiment_group}" and \
                                             experiment_id = "{experiment_id}" \
                                       '))
    if num_rows_to_delete > max_delete:
        print(f'Attempting to delete more than {max_delete} rows from table: trials. Exiting without deleting')
    else:
        if db.query(f'DELETE from trials WHERE trial_id = "{trial_id}" and \
                     mouse_id = "{mouse_id}" and start_time = "{start_time}" and \
                     stop_time = "{stop_time}" and behavior_file = "{behavior_file}" and \
                     tSeries_path = "{tSeries_path}" and experiment_group = "{experiment_group}" and \
                     experiment_id = "{experiment_id}" \
                 '):
            print(f'Deleted {num_rows_to_delete} from trials')
    db.disconnect()

def delete_row_from_trial_attributes(trial_id):
    db = ExperimentDatabase()
    max_delete = 1
    num_rows_to_delete = len(db.select_all(f'SELECT * FROM trial_attributes WHERE trial_id = {trial_id}'))
    if num_rows_to_delete > max_delete:
        print(f'Attempting to delete more than {max_delete} rows from table: trial_attributes. Exiting without deleting')
    else:
        if db.query(f'DELETE from trial_attributes WHERE trial_id = {trial_id}'):
            pass
            #print(f'Deleted {num_rows_to_delete} from trial_attributes')
    db.disconnect()
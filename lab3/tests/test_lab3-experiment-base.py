# possible TODO: save AnalysisResults from the various Experiment objects and
# and have the unit tests load and compare to freshly computed analyses
from lab3.experiment.base import _BaseExperiment, BehaviorExperiment, ImagingExperiment
from utils import *
from datetime import datetime
import numpy
import pytest
import sys

path_prefix = '/root/code/lab3/tests'

@pytest.mark.xfail # test expected to fail (currently can't connect to database)
@pytest.mark.lab3
@pytest.mark.lab3_experiment
@pytest.mark.lab3_experiment_base
@pytest.mark.functions
def test_fetch_trials():
        received = fetch_trials(project_name='test', mouse_name='test')
        assert received == [12707, 12709, 12708, 14295, 14299, 20908]
        received = fetch_trials('tSeriesDirectory', mouse_name='test')
        assert received == []
        received = fetch_trials(project_name='james', mouse_name='jbp027', 
                                experimentType='delayed_associative_memory', 
                                session=0, condition=0, start_time='2018-10-05-17h53m45s')
        assert received == [16716]
        
@pytest.fixture
def baseExperiment():
    return _BaseExperiment()

@pytest.mark.lab3
@pytest.mark.lab3_experiment
@pytest.mark.lab3_experiment_base
@pytest.mark.BaseExperiment
def test__BaseExperiment(baseExperiment):
        assert baseExperiment != None
        assert baseExperiment.trial_id == None
        assert baseExperiment._trial_info == {'trial_id': None, 'mouse_id': ''}
        assert baseExperiment.mouse_id == ''
        
@pytest.fixture
def behaviorExperiments():
    return [BehaviorExperiment(12707), BehaviorExperiment(16716)]

@pytest.fixture
def behaviorExperiments_loaded():
    with open('behaviorExperiments.pickle', 'rb') as f:
        return pickle.load(f)

@pytest.mark.xfail # can't connect to database
@pytest.mark.lab3
@pytest.mark.lab3_experiment
@pytest.mark.lab3_experiment_base
@pytest.mark.BehaviorExperiment
def test_BehaviorExperiment(behaviorExperiments, behaviorExperiments_loaded):
        assert all(behaviorExperiments) # check none of the behavior experiments are None
        assert all(behaviorExperiments_loaded)
        for exp, exp_loaded in zip(behaviorExperiments, behaviorExperiments_loaded):
            assert compare_dicts(exp.identifiers(), exp_loaded.identifiers())
            assert compare_dicts(exp.behavior_data, exp_loaded.behavior_data)
            # some BehaviorExperiment objects complain when you call format_behavior_data()
            try:
                assert compare_dicts(exp.format_behavior_data(), exp_loaded.format_behavior_data())
            except KeyError:
                pass

@pytest.fixture
def imagingExperiments_loaded():
    file_path = os.path.join(path_prefix, 'imageExperiments.pickle')
    with open(file_path, 'rb') as f:
        return pickle.load(f)

@pytest.fixture
def db():
    db = ExperimentDatabase()
    yield db
    db.disconnect()
    
@pytest.mark.xfail # can't connect to database
@pytest.mark.lab3
@pytest.mark.lab3_experiment
@pytest.mark.lab3_experiment_base
@pytest.mark.ImagingExperiment
def test_ImagingExperiment(imagingExperiments_loaded, db):
    trial_id = 999999
    sima_path = '/data3/jack/2p/jb130/20191124/jb130_20191124_0f193278bcf8ac5_1-001/jb130_20191124_0f193278bcf8ac5_1-001.2564997a.sima'
    delete_row_from_trial_attributes(trial_id)
    imagingExperiments = [ImagingExperiment(trial_id, sima_path=sima_path, force_pairing=True, store=True)]
    database_record = db.select_all("SELECT * FROM trial_attributes WHERE trial_id = 999999")[0]['value']
    
    assert database_record == sima_path
    
    with pytest.raises(AssertionError):
        img_exp_1 = ImagingExperiment(999999, 'fake/path.sima', force_pairing=True, store=True)
    with pytest.raises(ValueError):
        img_exp_1 = ImagingExperiment(999999, sima_path)
        
    for exp, exp_loaded in zip(imagingExperiments, imagingExperiments_loaded):
        assert exp.signals_file().filename == exp.signals_file().filename
        # TODO: test signals()
        # TODO: test calculate_dfof()
        # TODO: test infer_spikes()
        # TODO: test format_behavior_data()
        # TODO: test velocity()
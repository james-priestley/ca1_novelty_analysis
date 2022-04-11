from lab3.experiment.base import fetch_trials, ImagingExperiment


trial_ids = fetch_trials(
    project_name='steph', experimentType='contextAlternationAB_abbreviated')

for trial in trial_ids:
    try:
        expt = ImagingExperiment(trial)
        ds = expt.suite2p_imaging_dataset

        print('Running extraction for %s' % expt.sima_path)
        ds.extract(register=True, sparse_mode=True)

    except AttributeError:
        pass

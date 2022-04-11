import sys
if sys.version_info[0] < 3:
    raise Exception("lab3 may only be imported in Python 3")


from lab3.experiment.base import \
    fetch_trials, BehaviorExperiment, ImagingExperiment

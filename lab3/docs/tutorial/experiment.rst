========================
Using Experiment objects
========================

This tutorial will give a quick survey of the experiment module. The major 
tools provided here are in ``base``, which provides a set of objects to 
represent the most common types of experiments done in the lab. We'll point out 
the main functionality, and point out the major departures from the old repo's 
experiment class when necessary.

By the end of this tutorial, you will know how to initialize 
``BehaviorExperiment`` and ``ImagingExperiment`` objects, how to pair imaging 
data with trials in the experiment database, and how to access the raw behavior 
and imaging data through the experiment object properties and methods.

Using ``BehaviorExperiment``
----------------------------

The basic experiment types are locarted in ``lab3.experiment.base``.
``BehaviorExperiment`` is the base class for representing experiments that have
behavior data stored in the sql database. Instances are initialized by simply
passing the trial ID to the constructor. You can read about searching the
database for the IDs of specific trials in the database tutorial. Here we
initialize an experiment object::

    from lab3.experiment.base import BehaviorExperiment
    
    trial_id = 9001
    expt = BehaviorExperiment(trial_id)
    
This object behaves similarly to the old repo's ``dbExperimentClass`` for
accessing and modifying database attributes (though this may change slightly in
the near future). 

Perhaps the most common task is to access the behavior data associated with
this experiment. We have two methods of doing this::

    # this property returns the unformatted behavior dictionary stored in the
    # behavior pkl file on disk
    beh_dict = expt.behavior_data
    
    # this method returns the formatted behavior dictionary, where interval
    # variables are converted to indicator variables at a fixed sampling
    # rate 
    formatted_beh_dict = expt.format_behavior_data()
    
    # we can specify a custom sampling interval or trim data from the edges
    formatted_beh_dict = expt.format_behavior_data(sample_interval=0.1)
    formatted_beh_dict = expt.format_behavior_data(
        sample_interval=0.1, discard_pre=10, discard_post=10)
    formatted_beh_dict = expt.format_behavior_data(
        sample_interval=0.1, discard_pre='first_lap')

A velocity variable is included in the dictionary returned by 
``format_behavior_data()``, and we may also access it via ``velocity()``. Both
methods take a ``sigma`` parameter to control the degree of smoothing::

    # from the behavior dictionary
    velo = expt.format_behavior_data(sigma=0.1)['velocity']
    
    # from the method
    velo = expt.velocity(sigma=0.1)
    
    # velocity() takes any of the formatting parameters of 
    # format_behavior_data()
    velo = expt.velocity(sigma=0.1, discard_pre=10, sampling_interval=0.5)
    
TODO : add a section on modifying database attributes via 
``BehaviorExperiment``.


Using ``ImagingExperiment``
---------------------------

``ImagingExperiment`` is the basic class for representing experiments that
have both behavior data in the database and associated imaging data. There are
two ways to initialize an instance, depending on whether or not the database
has a stored record of where the associated imaging data is located. So the
constructor provides a convenient method for completing this pairing and
modifying the database for posterity. Experiments that are already paired can
simply be initialized with the trial ID alone.

Let's assume the experiment is in the database but is unpaired::

    from lab3.experiment.base import BehaviorExperiment, ImagingExperiment
    
    trial_id = 9001
    
    # this will work
    expt = BehaviorExperiment(trial_id)
    
    # this will break! 
    expt = ImagingExperiment(trial_id)
    
    # here is the path to the imaging data"
    sima_path = "/path/to/sima/directory.sima"
    
    # now we can initialize the object
    expt = ImagingExperiment(trial_id, sima_path=sima_path)
    
    # the above allows us to initialize the object now, but the results are
    # not automatically stored in the database. to do so, we must set
    # store=True
    expt = ImagingExperiment(trial_id, sima_path=sima_path, store=True)
    
    # we can also change the sima_path during initialization, and choose
    # whether to update the database
    new_sima_path = "/path/to/new/sima/directory.sima"
    
    # this does not edit the database
    expt = ImagingExperiment(trial_id, sima_path=new_sima_path, 
                             force_pairing=True)
    
    # this edits the database
    expt = ImagingExperiment(trial_id, sima_path=new_sima_path, 
                             force_pairing=True, store=True)
                             
The above example illustrates an import divergence from the standards of the
old repository. Here we unambiguously pair database records with 
*sima directories*, not the folder that contains them (previously an attribute
called ``tSeries_path``). Importantly this means that any experiments that
were paired with imaging data in the old format will have to be re-paired
(this will not affect the ``tSeries_path`` attribute, so you may continue to
use the old repository).

Before attempting to use then new imaging classes, it is also necessary to run 
``/scripts/update_h5.py`` on your imaging folder. This will find all h5
imaging datasets in your folder and (1) add some metadata to be permanently
stored in the h5 (e.g. the frame period) rather than calculated ad hoc
as before, and (2) wrap any naked h5 datasets with a sima directory (existing
sima folders are left untouched). This update currently only works for Prairie
datasets. We anticipate in the future that this step will be added to the
``prairie2h5.py`` script (and other conversion scripts). 

Properties and methods of ``ImagingExperiment``
-----------------------------------------------

Initializing an ``ImagingExperiment`` object gives us easy access to many
properties::

    dur = expt.frame_period
    fs = expt.frame_rate 
    param_dict = expt.imaging_parameters
    
    # since each experiment is paired with a sima folder, we can easily access
    # the underlying sima ImagingDataset object
    sima_ds = expt.imaging_dataset
    sequences = expt.imaging_dataset.sequences
    
    # we can also get a Suite2pImagingDataset, which will allow us to run the
    # Suite2p extraction code (see the Suite2p tutorial)
    s2p_ds = expt.suite2p_imaging_dataset
    
    
We can also run all of our basic signal processing steps directly through the
experiment object, rather than running scripts. This is covered in detail in
the signal processing tutorial. Here we summarize simply how to access the
signals::

    # this returns the path to the signals.h5 that stores time series data
    path = expt.signals_path
    
    # this returns the HDFStore object containing the signal traces. this is
    # read-only by default, but you can pass the usual arguments to the 
    # constructor 
    signals_file = expt.signals_file(mode='a')
    
    # we can retrieve specific signal entries from the file
    dfof = expt.signals(channel='Ch2', label='suite2p', signal_type='dfof)
    
    # we can get the ROI objects for a given label like so:
    rois = expt.rois(label='suite2p')
    
    # we can delete all ROI and signal records for a given label
    # if you do not set do_nothing=False, it will simply print the changes
    # that would be made
    expt.delete_roi_list(channel='Ch2', label='suite2p', do_nothing=False)

``ImagingExperiment`` inherits the previous behavior methods discussed from 
``BehaviorExperiment``, but modifies their functionality slightly to permit
easy synchronization of the imaging and behavior data::

    # we can set the sampling_interval and trim criteria to match the
    # imaging data via image_sync. Note these are True by default for an
    # ImagingExperiment
    formatted_beh_dict = expt.format_behavior_data(image_sync=True)
    velo = expt.velocity(image_sync=True)
    
    # signals() also takes a trim_to_behavior parameter, which is also True by
    # default
    dfof = expt.signals(channel='Ch2', label='suite2p', signal_type='dfof,
                        trim_to_behavior=True)


Using ``ImagingOnlyExperiment``
-------------------------------

Have an imaging dataset that isn't associated with behavior data? Or want to
start processing your signals without figuring out where you left your tdml
files months ago? You can get most of the functionality of 
``ImagingExperiment`` using ``ImagingOnlyExperiment``::

    from lab3.experiment.base import ImagingOnlyExperiment
    
    trial_id = 9001
    sima_path = "/path/to/unpaired/sima/directory.sima"
    
    # initialize the object
    expt = ImagingOnlyExperiment(trial_id, sima_path=sima_path)
    
Using this object you can access imaging data, run signals analysis, etc...
Obviously behavior methods and properties are unavailable.







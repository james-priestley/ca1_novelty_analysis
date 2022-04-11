========================
Signal processing basics
========================

Suppose you have a list of experiments that you would like to process::

    from lab3.experiment.base import ImagingExperiment, fetch_trials
    
    trial_ids = fetch_trials(project_name='awesome', mouse_name='tolstoy')
    expt_list = [ImagingExperiment(trial_id) for trial_id in trial_ids]
    
We'll assume that each of these experiments has signals imported, i.e. in
``expt.signal_file()`` there is a ``raw`` signal key. We'll assume that user
``awesome`` used Suite2p with the default settings, so there will be two signal
records: ``/Ch2/suite2p/raw`` and ``/Ch2/suite2p/npil``. 

We can access these signals easily::
    
    expt = expt_list[0]
    raw = expt.signals(label='suite2p', signal_type='raw')
    
Here ``raw`` is a pandas dataframe indexed by the ROI labels.

Calculating DFOF
----------------

The ``lab3.signal.dfof`` module gives us a number of strategies for calculating
dF/F from raw signals. Since we are considering Suite2p extractions here, which
include also an estimate of the neuropil, we'll use the ``Suite2pDFOF`` 
strategy::

    from lab3.signal.dfof import Suite2pDFOF

All subclasses of ``DFOFStrategy`` share a common interface. So we could 
calculate dF/F like so::

    # initialize the strategy objects with any custom parameters we want
    # the parameters available may change depending on the specific strategy
    strategy = Suite2pDFOF(constant_denominator=True)
    
    # load our signals
    raw = expt.signals(label='suite2p', signal_type='raw')
    npil = expt.signals(label='suite2p', signal_type='npil')
    
    # calculate dfof
    dfof = strategy.calculate(raw, npil)
    
We can also use methods of ``Imagingexperiment`` to calculate dF/F using an
arbitrary strategy, and automatically store the results in the experiment's
signals file::

    # let's try a different strategy
    from lab3.signal.dfof import JiaDFOF
    
    for expt in expt_list:
        
        dfof = expt.calculate_dfof(JiaDFOF(), channel='Ch2', label='suite2p', 
                                   overwrite=True)
        
        # now we can retrieve the signals via the signals() method
        dfof = expt.signals(channel='Ch2', label='suite2p', signal_type='dfof')

Spike inference
---------------
TODO but the gist is similar to the above 
        
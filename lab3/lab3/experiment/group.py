"""A collection of classes for representing groups of experiments."""

from functools import reduce
from collections import defaultdict

from lab3.core import Group
from lab3.experiment.base import (fetch_trials, _BaseExperiment,
                                  BehaviorExperiment, ImagingExperiment)


class ExperimentGroup(Group):

    """Class for representing groups of Experiment objects. These can be
    instantiated directly from an iterable of experiment objects, or via
    helper methods.

    Parameters
    ----------
    iterable_of_expts : iterable of _BaseExperiment instances
        Experiments to include in group
    name : str, optional
        Name for ExperimentGroup object
    parallelize : bool, optional
        Whether to parallelize `apply` calls, defaults to False.

    Examples
    --------
    From experiment objects:

    >>> list_of_expts = [BehaviorExperiment(trial_id=1),
                         BehaviorExperiment(trial_id=2)]
    >>> expt_grp = ExperimentGroup(list_of_expts)

    From trial_ids via helper method:

    >>> expt_grp = ExperimentGroup.from_trial_ids(
            [1, 2], expt_class=BehaviorExperiment
        )

    From database query via helper method:

    >>> expt_grp = ExperimentGroup.from_database(
            project_name='world_domination',
            mouse_name='Pinky'
        )

    An ExperimentGroup instance can be transformed into a Cohort, which is a
    group of ExperimentGroups that organizes the experiments by mouse:

    >>> cohort = expt_grp.to_cohort()
    >>> cohort[0]  # this returns a Mouse object (an ExperimentGroup)
    """

    def __init__(self, iterable_of_expts, **kwargs):
        assert all([isinstance(i, _BaseExperiment)
                    for i in iterable_of_expts]), \
            "All elements must be an instance of an experiment class!"
        super().__init__(iterable_of_expts, **kwargs)

    @classmethod
    def from_trial_ids(cls, trial_ids, expt_class=ImagingExperiment,
                       name='unnamed', parallelize=False, **kwargs):
        """Helper method for constructing ExperimentGroup objects from a list
        of database trial IDs.

        Parameters
        ----------
        trial_ids : list of ints
            Integer database trial_id for each experiment to include in group
        expt_class : _BaseExperiment, optional
            Type of Experiment class to generate for each element of the
            ExperimentGroup. Must be a subclass of _BaseExperiment. Defaults to
            ImagingExperiment
        name : str, optional
            Name for ExperimentGroup object
        parallelize : bool, optional
            Whether to parallelize `apply` calls, defaults to False.
        **kwargs : keyword arguments passed to expt_class initializer
        """
        return cls([expt_class(tid, **kwargs) for tid in trial_ids], name=name,
                   parallelize=parallelize, item_class=expt_class)

    @classmethod
    def from_database(cls, expt_class=ImagingExperiment,
                      name='unnamed', parallelize=False, **db_kwargs):
        """Helper method for constructing ExperimentGroup objects by querying
        the database for the desired experiments.

        Parameters
        ----------
        project_name, mouse_name, experimentType : str, optional
            Return experiments with these property names. Arguments left as
            None (default) are not used to filter.
        expt_class : _BaseExperiment, optional
            Type of Experiment class to generate for each element of the
            ExperimentGroup. Must be a subclass of _BaseExperiment. Defaults to
            ImagingExperiment
        name : str, optional
            Name for ExperimentGroup object
        parallelize : bool, optional
            Whether to parallelize `apply` calls, defaults to False.
        **db_kwargs
            Additional keyword arguments are passed directly to
            lab3.experiment.base.fetch_trials.

        See also
        --------
        lab3.experiment.base.fetch_trials
        """
        trial_ids = fetch_trials(**db_kwargs)
        return cls.from_trial_ids(trial_ids, expt_class=expt_class,
                                  name=name, parallelize=parallelize)

    @property
    def trial_ids(self):
        return [e.trial_id for e in self]

    def to_cohort(self, name=None):
        """Convert this ExperimentGroup into a Cohort (a group of Mice -- so a
        group of ExperimentGroups)."""

        mouse_dict = defaultdict(list)
        for e in self:
            mouse_dict[e.mouse_id].append(e)
        mice = [Mouse(expts, item_class=self.item_class,
                      parallelize=self.parallelize)
                for mouse_id, expts in mouse_dict.items()]

        if name is None:
            name = self.name

        return Cohort(mice, name=name)

    def to_list(self):
        return [e for e in self]

    def __add__(self, other):
        return self.__class__([e for e in self] + [e for e in other],
                              name=self.name, parallelize=self.parallelize)


class Mouse(ExperimentGroup):

    """Mouse objects are ExperimentGroup objects that contain experiments
    from the same mouse.

    Parameters
    ----------
    iterable_of_expts : iterable of _BaseExperiment instances
        Experiments to include in group
    name : ignored
        The group name is automatically set to be the mouse name.
    parallelize : bool, optional
        Whether to parallelize `apply` calls, defaults to False.

    Examples
    --------
    # load all experiments for Pinky

    >>> m = Mouse.from_database('Pinky', experiment_type='random_foraging')

    Note the default experiment class used by Mouse (and ExperimentGroup) is
    ImagingExperiment, so if you haven't paired all experiments with sima_paths
    yet, you should first load the Mouse using BehaviorExperiment:

    >>> from lab3 import BehaviorExperiment
    >>> m = Mouse.from_database('Pinky', expt_class=BehaviorExperiment)

    The Mouse class is endowed with a convenience method to pair the behavior
    experiments with their corresponding imaging datasets

    >>> m = m.pair_imaging_data(imaging_path="/data9/world_domination")
    >>> m[0]  # this is now an ImagingExperiment (if pairing was successful)

    See also
    --------
    lab3.experiment.group.ExperimentGroup
    """

    def __init__(self, iterable_of_expts, **kwargs):

        mouse_ids = [e.mouse_id for e in iterable_of_expts]
        assert len(set(mouse_ids)) == 1, \
            "All experiments must have the same mouse_id!"

        # set the name (or should we use the ID number instead?)
        kwargs['name'] = iterable_of_expts[0].mouse_name
        super().__init__(iterable_of_expts, **kwargs)

    @classmethod
    def from_database(cls, mouse_name, expt_class=ImagingExperiment,
                      parallelize=False, **db_kwargs):
        """Helper method for constructing Mouse objects by querying the
        database for the desired experiments.

        Parameters
        ----------
        mouse_name : str
            Mouse to return
        project_name, experimentType : str, optional
            Return experiments with these property names. Arguments left as
            None (default) are not used to filter.
        expt_class : _BaseExperiment, optional
            Type of Experiment class to generate for each element of the
            ExperimentGroup. Must be a subclass of _BaseExperiment. Defaults to
            ImagingExperiment
        name : ignored
        parallelize : bool, optional
            Whether to parallelize `apply` calls, defaults to False.
        **db_kwargs
            Additional keyword arguments are passed directly to
            lab3.experiment.base.fetch_trials.

        See also
        --------
        lab3.experiment.base.fetch_trials
        lab3.experiment.group.ExperimentGroup
        """
        trial_ids = fetch_trials(mouse_name=mouse_name, **db_kwargs)
        return cls.from_trial_ids(trial_ids, expt_class=expt_class,
                                  parallelize=parallelize)

    def pair_imaging_data(self, *args, do_nothing=False, **kwargs):
        """Use the Hungarian algorithm to find the optimal pairings between
        behavior experiments and imaging datasets, using the start time
        difference between the behavior tdml and Prairie xml files as the cost
        function.

        Parameters
        ----------
        imaging_path : str
            Locate Prairie xmls/sima folders below this folder
        max_dt : float, optional
            Maximum timestamp difference (in seconds) between behavior tdml and
            Prairie xml files. Defaults to 60
        force_pairing : bool, optional
            Whether to overwrite existing sima pairs. Defaults to False
        keywords : iterable of strs, optional
            Additional keywords to identify in sima_paths
        do_nothing : bool, optional
            Find pairs but don't update the database. Defaults to False
        verbose : bool, optional
            Display extra information. Defaults to False
        """

        assert all([isinstance(e, BehaviorExperiment) for e in self]), \
            "Cannot find imaging/behavior pairs unless all members of " \
            + "ExperimentGroup are instances of BehaviorExperiment"

        from lab3.experiment._pairing import pair_mouse_experiments
        pair_mouse_experiments(self, *args, do_nothing=do_nothing, **kwargs)

        if not do_nothing:
            # Note this may still break if some experiments were unpaired
            return Mouse.from_trial_ids([e.trial_id for e in self],
                                        expt_class=ImagingExperiment,
                                        parallelize=self.parallelize)
        else:
            return self


class Cohort(Group):

    """Cohort objects are groups of Mouse objects.

    Examples
    --------
    >>> cohort[0]  # this is a Mouse (ExperimentGroup)
    >>> cohort[0][0]  # this is an Experiment object
    """

    def __init__(self, iterable_of_mice, **kwargs):
        assert all([isinstance(m, Mouse) for m in iterable_of_mice]), \
            "All elements must be an instance of the Mouse class!"
        super().__init__(iterable_of_mice, **kwargs)

    @classmethod
    def from_ExperimentGroup(cls, expt_grp, name='unnamed'):
        return expt_grp.to_cohort(name=name)

    @classmethod
    def from_database(cls, name='unnamed', **kwargs):
        expt_grp = ExperimentGroup.from_database(**kwargs)
        return cls.from_ExperimentGroup(expt_grp, name=name)

    @property
    def trial_ids(self):
        trial_ids = []
        for grp in self:
            trial_ids.extend(grp.trial_ids)
        return trial_ids

    def to_list(self):
        return reduce(lambda x, y: x + y, [grp.to_list() for grp in self])


class MouseGroup(Cohort):

    """This is simply an alias for Cohort"""

    pass


class CohortGroup(Group):

    """CohortGroup objects are groups of Cohorts.

    Examples
    --------
    >>> cohort[0]  # this is a Cohort
    >>> cohort[0][0]  # this is a Mouse (ExperimentGroup)
    >>> cohort[0][0][0]  # this is an Experiment object
    """

    def __init__(self, iterable_of_cohorts, **kwargs):
        assert all([isinstance(m, Cohort) for m in iterable_of_cohorts]), \
            "All elements must be an instance of the Cohort class!"
        super().__init__(iterable_of_cohorts, **kwargs)

    @property
    def trial_ids(self):
        trial_ids = []
        for grp in self:
            trial_ids.extend(grp.trial_ids)
        return trial_ids

    def to_list(self):
        return reduce(lambda x, y: x + y, [grp.to_list() for grp in self])

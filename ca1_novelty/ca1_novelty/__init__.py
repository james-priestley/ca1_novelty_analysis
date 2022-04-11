# ----------------------- Mice used in main dataset ------------------------- #

PC_MICE = [
    'jbp_vr_031',
    'jbp_vr_032',
    'jbp_vr_033',
    'jbp_vr_034',
    'jbp_vr_035',
    'jbp_vr_036'
]


LC_PILOT_MICE = [
    'jbp_vr_037',  # pilot data only! bad behavior
    'jbp_vr_040'   # pilot data only! bad behavior
]


# --------------- Convenience experiment loading functions ------------------ #

def load_context_switch_experiments():

    from lab3.experiment.group import Cohort
    from ca1_novelty.experiment import PlaceCellContextExperiment

    return Cohort.from_database(
        project_name='james',
        experimentType='context_switch',
        day=[0, 1, 2],
        mouse_name=PC_MICE,
        expt_class=PlaceCellContextExperiment,
    )


def load_single_context_experiments():

    from lab3.experiment.group import Cohort
    from ca1_novelty.experiment import PlaceCellContextExperiment

    return Cohort.from_database(
        project_name='james',
        experimentType='single_context',
        mouse_name=PC_MICE,
        expt_class=PlaceCellContextExperiment,
    )


def load_LC_pilot_data():

    pass

"""Common metrics calculate from spatial tuning curves and place fields"""


import numpy as np

# Should these generically take single or multiple tuning curves/place fields?
# Probably just single, and use pandas to apply row-wise on the corresponding
# dataframes


def skaggs_information(tuning_curve, occupancy=None):
    """Skaggs spatial info"""

    if occupancy is None:
        occupancy = np.ones(tuning_curve.shape)

    assert all(occupancy >= 0), "Occupancy must be non-negative!"
    occupancy /= occupancy.sum()

    fr_ratio = tuning_curve / tuning_curve.mean()
    return np.nan_to_num(occupancy * fr_ratio * np.log2(fr_ratio)).sum()


def entropy(f):
    """Shannon entropy, treating the tuning curve as a pseudo-probability mass
    (i.e., normalized so that it sums to 1)."""
    denom = f.sum(axis=-1)
    if len(f.shape) == 2:
        denom = denom.reshape(-1, 1)
    p = f / denom
    return -1 * (p * np.nan_to_num(np.log2(p))).sum(axis=-1)


def tuning_score(f):
    """Spatial tuning score, calculated as the KL divergence between the
    normalized tuning curve and the discrete uniform distribution."""
    return np.nan_to_num(np.log2(f.shape[-1]) - entropy(f))


def spatial_reliability():
    return np.nan


def spatial_tuning_gain():
    raise NotImplementedError


def place_field_gain():
    raise NotImplementedError


def first_lap():
    """Return the lap number for the first time the cell fired in its place
    field"""
    return np.nan

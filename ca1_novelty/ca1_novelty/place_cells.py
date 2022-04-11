"""Module for place cell analysis code"""

import numpy as np
import pandas as pd

from platz.field import PlaceField1D

from ca1_novelty.xcorr import center_of_mass


class PlaceField(PlaceField1D):

    """
    Parameters
    ----------
    start, stop : in
    n : int, optional
        Number of active laps in window to qualify as the place field start
        Defaults to 3 laps
    window : int, optional
        Lap window size to consider when looking for consistent in-field
        firing. Defaults to 3 laps
    margin : int, optional
        Expand the place field boundaries by this many bins on each side
        (to account for possible spatial shifts due to BTSP, for example).
        Defaults to 5
    """

    def analyze_btsp(self, **kwargs):
        """
        """

        first_lap_activity = \
            self.raster[self.first_lap, self.start_bin:self.stop_bin]
        first_lap_max = np.max(first_lap_activity)
        first_lap_pos = center_of_mass(first_lap_activity) + self.start_bin

        remaining_activity = np.mean(
            self.raster[(self.first_lap+1):][:, self.start_bin:self.stop_bin],
            axis=0)
        remaining_max = np.max(remaining_activity)
        remaining_pos = center_of_mass(remaining_activity) + self.start_bin

        results = {
            'first_lap': self.first_lap,
            'first_lap_pos': first_lap_pos,
            'remaining_pos': remaining_pos,
            'gain': first_lap_max / remaining_max,
            'drift': first_lap_pos - remaining_pos,
            'width': self.stop - self.start,
        }

        return pd.Series(results)

    def relative_lap_centers(self, **kwargs):
        in_field_raster = self.raster[
            self.first_lap:, self.start_bin:self.stop_bin]
        in_field_centers = center_of_mass(in_field_raster) + self.start_bin
        in_field_centers -= np.nanmedian(in_field_centers)
        return in_field_centers

    def centered_raster(self, n_laps=10, half_window=25):
        truncated_raster = self.raster[self.first_lap:self.first_lap + n_laps]
        com = center_of_mass(
            truncated_raster[1:].mean(axis=0)[self.start_bin:self.stop_bin])
        com = np.round(com) + self.start_bin
        shift = int(self.n_bins / 2 - com)
        truncated_raster = np.roll(truncated_raster, shift, axis=1)

        if truncated_raster.shape[0] < n_laps:
            truncated_raster = np.vstack(
                [truncated_raster,
                 np.zeros((n_laps - truncated_raster.shape[0],
                           truncated_raster.shape[1]))]
            )

        start = int(self.n_bins / 2 - half_window)
        stop = int(self.n_bins / 2 + half_window)

        return truncated_raster[:, start:stop]

    def nmf_field_info(self, **kwargs):
        stats = self.analyze_btsp()
        stats['raster'] = self.centered_raster(**kwargs)
        return stats


class PlacePopulation(list):

    pass

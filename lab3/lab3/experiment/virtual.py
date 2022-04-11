"""Classes that extend the functionality of experiment objects to handle
operations specific to experiments collected on the virtual reality rigs."""

import os

import numpy as np
from .base import BehaviorExperiment, ImagingExperiment


class VirtualBehaviorMixin:

    @property
    def vr_context_names(self):
        """Get the name of variables in self.behavior_data that correspond
        to virtual reality contexts."""

        scene_names = []
        for ctx_dict in \
                self.behavior_data['__trial_info']['contexts'].values():

            if 'vr_file' in ctx_dict:
                scene_names.append(os.path.splitext(ctx_dict['vr_file'])[0])
            elif 'scene_name' in ctx_dict:
                scene_names.append(ctx_dict['scene_name'])
            else:
                # not a VR context. any other cases we should handle?
                pass

        return sorted(scene_names)

    def vr_context_vars(self, **kwargs):
        return {name: self.format_behavior_data(**kwargs)[name]
                for name in self.vr_context_names}

    def iti(self, **kwargs):
        """Use context variables to determine sample bins that fall during
        the inter-trial-interval periods"""
        iti = np.sum(list(self.vr_context_vars(**kwargs).values()),
                     axis=0) == 0

        # correct the right boundaries of the ITIs
        lap_starts = np.where(np.diff(iti.astype('int')) == -1)[0] + 1
        for lap_num, s in enumerate(lap_starts):
            margin = np.argmin(self.discrete_position(**kwargs)[s:s + 30])
            iti[s:s + margin] = True

            # there is an unhandled edge-case here -- what if the animal runs
            # really fast at the start of the lap and so misses pos 0, but
            # then back-pedals?

        # iti_ends = np.where(np.diff(iti.astype('int')) == -1)[0] + 1
        # for end in iti_ends:
        #     # look ahead 20 samples, find the first sample that drops below 10
        #     pos_snippet = self.discrete_position(**kwargs)[end:end + 20]
        #     extension = np.where(pos_snippet < 10)[0][0]  # + 1
        #     iti[end:end + extension] = True

        return iti

    def sample_vr_context_id(self, **kwargs):
        """Get a list with elements corresponding to the context name on each
        sample."""
        sample_id = np.asarray([None] * len(self.iti(**kwargs)))
        for key, value in self.vr_context_vars(**kwargs).items():
            sample_id[value] = key
        sample_id[self.iti(**kwargs)] = None
        return sample_id

    def lap_vr_context_id(self, **kwargs):
        """Get a list with elements corresponding to the context name on each
        lap"""
        sample_ids = self.sample_vr_context_id(**kwargs)[~self.iti(**kwargs)]

        true_lap_starts = np.concatenate(
            [[0], np.where(np.diff(
                self.laps(**kwargs)[~self.iti(**kwargs)]))[0] + 2]
            )

        return sample_ids[true_lap_starts]

    def valid_samples(self, velocity_threshold=5, **kwargs):
        """Make a boolean variable that is True at each sample that exceeds
        the velocity threshold and is not during an ITI period

        Parameters
        ----------
        velocity_threshold : float, optional
            Minimum velocity for inclusion, in cm/sec. Defaults to 5.
        """
        return (~ self.iti(**kwargs)) \
            & (self.velocity(**kwargs) > velocity_threshold)

    #
    # def format_behavior_data(self, **kwargs):
    #     """Adds ITI and valid_samples variables to the behavior_data
    #     dictionary"""
    #     raise NotImplementedError


class VirtualImagingMixin:

    pass


class VirtualBehaviorExperiment(VirtualBehaviorMixin, BehaviorExperiment):

    pass


class VirtualImagingExperiment(VirtualImagingMixin, VirtualBehaviorMixin,
                               ImagingExperiment):

    pass

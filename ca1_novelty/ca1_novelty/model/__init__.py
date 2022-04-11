import numpy as np
import pandas as pd

from ca1_novelty.model.mouse import LinearMouse
from ca1_novelty.model.neuron import CA3InputEnsemble, CA1Neuron
from ca1_novelty.model.util import (
    poisson_spikes, reconvolve_spikes,
    infer_spikes, subsample_calcium
)


class BTSPSimulation:

    "Encapsulate all the analysis for a single simulation run here"

    def __init__(self, velocity, plateau_pos, plateau_duration=0.2, fs=10,
                 n_trials=16, dt=1e-3, mouse_kws={}, input_kws={},
                 output_kws={}, verbose=False):

        self.velocity = velocity
        self.fs = fs
        self.n_trials = n_trials
        self.dt = dt
        self.plateau_pos = plateau_pos
        self.plateau_duration = plateau_duration
        self.verbose = verbose

        # Set up network model
        if self.verbose:
            print('Setting up network...')
        input_model = CA3InputEnsemble(**input_kws)
        self.neuron = CA1Neuron(input_model, **output_kws)
        self.neuron.apply_btsp(plateau_pos, self.velocity)

        # Set up mouse and pre-generate behavior
        if self.verbose:
            print('Setting up behavior...')
        self.mouse = LinearMouse(**mouse_kws)
        self._generate_behavior()

        # Generate a square wave signal indicating the plateau on the first lap
        # if plateau_duration is not None
        # Otherwise all laps in simulation will be post-plasticity
        self._make_plateau_signal()

        # Generate instantaneous CA1 firing rate at each behavior_sample
        if self.verbose:
            print('Computing CA1 firing rate...')
        self.firing_rate = self.neuron.firing_rate(self.pos, self.dt,
                                                   self.plateau_signal)

    def _generate_behavior(self):
        self.mouse.reset()
        self.pos, self.laps = self.mouse.move_n_trials(
            self.n_trials, self.velocity, dt=self.dt)

    def _make_plateau_signal(self):
        if self.plateau_duration is not None:
            p = ((self.laps == 0)
                 & (self.pos > self.plateau_pos)
                 & (self.pos < (self.plateau_pos
                                + self.velocity * self.plateau_duration))
                 ).astype('float')
            p[self.laps > 0] = np.nan
            self.plateau_signal = p
        else:
            self.plateau_signal = None

    def _run(self, SNRs, calcium_kws, spike_kws):
        """Generate a single simulated spike train, and use this to produce
        surrogate calcium traces across the chosen SNRs"""

        np.random.seed()

        spikes = poisson_spikes(self.firing_rate, self.dt)
        calcium = reconvolve_spikes(spikes, **calcium_kws)

        expt = {}
        for snr in SNRs:

            while True:

                sub_calcium = subsample_calcium(calcium, snr, fs=self.fs,
                                                dt=self.dt)
                inferred_spikes = infer_spikes(sub_calcium, self.fs,
                                               **spike_kws) > 0

                # check that there are spikes on the first lap and the
                # remaining laps
                skip = int(1 / (self.dt * self.fs))
                sub_laps = self.laps[::skip]
                if (inferred_spikes[sub_laps == 0].sum() > 0) \
                        & (inferred_spikes[sub_laps > 0].sum() > 0):
                    break

            expt[snr] = pd.DataFrame({'sub_calcium': sub_calcium,
                                      'inferred_spikes': inferred_spikes})

        expt = pd.concat(expt, axis=1)
        ground_truth = pd.DataFrame({'spikes': spikes, 'calcium': calcium})

        return expt, ground_truth

    def run(self, n_sim, SNRs=[0.25, 0.5, 1, 2, 4, 8, 16],
            calcium_kws={'tau_d': 0.7, 'tau_r': 0.07}, spike_kws={'n_mad': 3}):
        """Main simulation function

        Parameters
        ----------
        n_sim : int
        SNRs : list of floats
        calcium_kws : dict
        spike_kws : dict

        Returns
        -------
        experimental : pd.DataFrame
        ground_truth : pd.DataFrame
        """

        ground_truth, experimental = {}, {}
        for n in range(n_sim):
            if self.verbose:
                print(n)
            experimental[n], ground_truth[n] = self._run(
                SNRs, calcium_kws, spike_kws)

        ground_truth = pd.concat(ground_truth, axis=1)

        # add behavior data to index
        ground_truth['position'] = self.pos
        ground_truth['laps'] = self.laps
        ground_truth['velocity'] = self.velocity
        ground_truth['time'] = np.arange(len(ground_truth)) * self.dt
        ground_truth.set_index(['velocity', 'time', 'position', 'laps'],
                               inplace=True)

        frame_skip = int(1 / (self.dt * self.fs))
        experimental = pd.concat(experimental, axis=1)

        # add behavior data to index
        experimental['position'] = self.pos[::frame_skip]
        experimental['laps'] = self.laps[::frame_skip]
        experimental['velocity'] = self.velocity
        experimental['time'] = \
            np.arange(len(experimental)) * self.dt * frame_skip
        experimental.set_index(['velocity', 'time', 'position', 'laps'],
                               inplace=True)

        return ground_truth, experimental

import numpy as np
import scipy.stats as sp
from scipy.signal import gaussian, convolve

from ca1_novelty.model.util import position_btsp, poisson_spikes


class PlaceField:

    """docs"""

    def __init__(self, center, sigma, max_rate):

        self.center = center
        self.sigma = sigma
        self.max_rate = max_rate

    def rate_by_position(self, x):
        return self.max_rate \
               * np.exp(- 1 * (x - self.center) ** 2 / (2 * self.sigma ** 2))


class CA3InputEnsemble:

    """"""

    def __init__(self, n=300, track_length=300, sigma=10, max_rate=50):

        self.n = n
        self.track_length = track_length
        self.sigma = sigma
        self.max_rate = max_rate

        self._make_place_fields()

    def _make_place_fields(self):
        self.place_fields = []
        for center in np.linspace(0, self.track_length, self.n):
            self.place_fields.append(
                PlaceField(center, self.sigma, self.max_rate))

    def firing_rate(self, x, dt):
        return np.array([pf.rate_by_position(x) * dt
                         for pf in self.place_fields])

    def spike(self, x, dt):
        r = self.firing_rate(x, dt)
        return np.where(np.random.uniform(0, 1, self.n) < (r))[0]

    def __len__(self):
        return self.n

    @property
    def field_centers(self):
        return np.array([pf.center for pf in self.place_fields])


class CA1Neuron:

    """For now the model works via weighted sum of pre-synaptic firing rates
    evaluated at the current position, but it would be straightforward to use
    presynaptic spikes with some weight decay dynamics. Using firing rates
    directly however allows us to compute the presynaptic current for all time
    steps at once."""

    def __init__(self, inputs, max_rate=30, plateau_scaling=3,
                 weight_prior=-10):

        self.inputs = inputs
        self.plateau_scaling = plateau_scaling
        self._initialize_weights(weight_prior)
        self.max_rate = max_rate

    def _initialize_weights(self, weight_prior):
        self.w = np.zeros(len(self.inputs))
        self.pre_w = np.zeros_like(self.w)
        if weight_prior is not None:
            self.pre_w[:] = weight_prior
            self.w[:] = weight_prior

        # self._synaptic_currents = np.zeros(len(self.inputs))

    def apply_btsp(self, x, velocity):
        self.w += position_btsp(x, self.inputs.field_centers, velocity)
        self.w = sp.zscore(self.w)

    def total_synaptic_current(self, x, dt, plateau_input=None):
        input_fr = self.inputs.firing_rate(x, dt=dt)

        if (plateau_input is None) or all(np.isnan(plateau_input)):
            return self.w @ input_fr
        elif np.isscalar(x):
            return self.pre_w @ input_fr
        else:
            current = self.w @ input_fr

            plateau_filter = ~np.isnan(plateau_input)
            pre_current = self.pre_w @ input_fr[:, plateau_filter]
            current[plateau_filter] = pre_current

            return current

    def firing_rate(self, x, dt, plateau_input=None):
        # TODO need to figure out how to handle pre/post plateau weight change
        total_current = self.total_synaptic_current(x, dt, plateau_input)

        scaling = np.log(self.max_rate) / np.nanmax(total_current)

        if plateau_input is not None:
            # lightly smooth plateau input
            k = gaussian(int(1 / dt), int(0.025 / dt))
            k /= np.sum(k)
            plateau_input = convolve(np.nan_to_num(plateau_input),
                                     k, mode='same')

            total_current += plateau_input \
                * self.plateau_scaling * np.nanmax(total_current)

        fr = np.exp(total_current * scaling)

        # if plateau_input is not None:
        #     fr += np.nan_to_num(plateau_input) * self.plateau_scaling \
        #           * np.nanmax(fr)
        return fr

    def spike(self, x, dt, plateau_input):
        return poisson_spikes(self.firing_rate(x, dt, plateau_input), dt)


# class PlateauSignal:
#
#     def __init__(self, plateau_x, duration=0.2, max_rate=45):
#
#         self.plateau_x = plateau_x
#         self.duration = duration
#         self.max_rate = max_rate
#
#     def firing_rate(self, x, dt):
#         pass


    # @property
    # def total_input(self):
    #     return self.synaptic_currents.sum() + \
    #         np.random.normal(self.bias, self.noise_sigma)

    # def run(self, x, dt):
    #     """Integrate the current inputs and decide whether to spike or not"""
    #     self._update_synaptic_currents(x, dt)
    #     return self._spike(dt)

    # def _active_synapses(self, x, dt):
    #     """Returns the indices of the synapses that currently received a
    #     pre-synaptic spike"""
    #     return self.inputs.spike(x, dt)

    # def _update_synaptic_currents(self, x, dt):
    #     # self._synaptic_currents -= self.synaptic_currents * (dt / self.syn_tau)
    #     # active = self._active_synapses(x, dt)
    #     # self._synaptic_currents[active] += self.w[active]
    #     self._synaptic_currents = self.w * self.inputs.firing_rate(x, dt)

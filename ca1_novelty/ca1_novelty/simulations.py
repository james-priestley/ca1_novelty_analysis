import numpy as np
from scipy.signal import gaussian


class PlaceFieldSimulator:

    def __init__(self, n_laps=30, n_bins=100, accum_prob=0.166, pf_prob=0.2,
                 noise_prob=0.2, pf_width=5, jitter=True):

        self.n_laps = n_laps
        self.n_bins = n_bins
        self.accum_prob = accum_prob
        self.pf_prob = pf_prob
        self.noise_prob = noise_prob
        self.pf_width = pf_width
        self.jitter = jitter

        self._gaussian_bump = gaussian(self.n_bins * 3, self.pf_width)
        self._track_start = int(self.n_bins * 1.5)

    def _draw_centroid(self):
        return np.floor(np.random.uniform(0, self.n_bins)).astype('int')

    def _draw_start_lap(self):
        return np.random.default_rng().geometric(self.accum_prob) - 1

    def _make_field(self, start_lap, centroid):
        r = np.zeros((self.n_laps, self.n_bins * 3))

        if (np.random.uniform(0, 1) < self.pf_prob) \
                & (start_lap < self.n_laps):
            prototype = np.roll(self._gaussian_bump, centroid)
            r[start_lap:] = np.stack((self.n_laps - start_lap) * [prototype])

        return r

    def make_field(self):
        pf = self._make_field(self._draw_start_lap(),
                              self._draw_centroid())

        # add out-of-field noise
        noise_laps = np.where(
            np.random.uniform(0, 1, self.n_laps) < self.noise_prob)[0]
        if len(noise_laps):
            pf[noise_laps] += np.stack(
                [np.roll(self._gaussian_bump, np.random.choice(self.n_bins))
                 for _ in range(len(noise_laps))]) * 0.5

        # add scaling noise
        pf *= np.random.normal(1, 0.05, self.n_laps).reshape(-1, 1)

        # add jitter
        if self.jitter:
            pf = np.stack(
                [np.roll(lap, np.random.choice(np.arange(-2, 3)))
                 for lap in pf]
            )

        return pf[:, self._track_start:(self._track_start + self.n_bins)]


class BTSPSimulator(PlaceFieldSimulator):

    def __init__(self, btsp_prob=0.5, max_shift=25, max_scale=3, **kwargs):

        self.btsp_prob = btsp_prob
        self.max_shift = max_shift
        self.max_scale = max_scale

        super().__init__(**kwargs)

    def _make_field(self, start_lap, centroid):
        r = np.zeros((self.n_laps, self.n_bins * 3))

        if (np.random.uniform(0, 1) < self.pf_prob) \
                & (start_lap < self.n_laps):
            prototype = np.roll(self._gaussian_bump, centroid)
            r[start_lap:] = np.stack((self.n_laps - start_lap) * [prototype])

            if np.random.uniform(0, 1) < self.btsp_prob:
                # shift and scale first lap
                r[start_lap] = np.roll(
                    r[start_lap],
                    np.random.randint(0, self.max_shift))
                r[start_lap] *= np.random.uniform(1, self.max_scale)

        return r


class DriftingSimulator(PlaceFieldSimulator):

    def __init__(self, drift_prob=0.5, max_slope=1, **kwargs):

        self.drift_prob = drift_prob
        self.max_slope = max_slope

        super().__init__(**kwargs)

    def _get_lap_drifts(self, start_lap):

        slope = np.random.uniform(0, self.max_slope)
        drifts = np.round(
            -1 * slope * np.arange(0, (self.n_laps - start_lap))).astype('int')

        return drifts

    def _make_field(self, start_lap, centroid):
        r = np.zeros((self.n_laps, self.n_bins * 3))

        if (np.random.uniform(0, 1) < self.pf_prob) \
                & (start_lap < self.n_laps):
            prototype = np.roll(self._gaussian_bump, centroid)
            r[start_lap:] = np.stack((self.n_laps - start_lap) * [prototype])

            if np.random.uniform(0, 1) < self.drift_prob:
                # shift and scale first lap
                r[start_lap:] = np.stack([
                    np.roll(lap, d) for lap, d in zip(
                        *[r[start_lap:], self._get_lap_drifts(start_lap)])
                ])

        return r

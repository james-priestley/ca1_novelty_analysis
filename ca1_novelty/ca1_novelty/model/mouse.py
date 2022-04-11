import copy

import numpy as np


class SimMouse:

    pass


class LinearMouse(SimMouse):

    """A mouse that moves in a bounded linear environment."""

    def __init__(self, track_length=500, init_x=0, iti_duration=None):

        self.track_length = track_length
        self.x = init_x
        self.iti_duration = iti_duration
        self._iti_counter = None
        self.trial_num = 0

    def _update_x(self, velocity, dt):
        dx = velocity * dt
        if self.iti_duration is None:
            self.x += dx
            self.x %= self.track_length
        else:
            # first check if we are in the iti
            if self._iti_counter is not None:
                self._iti_counter += dt
                if self._iti_counter >= self.iti_duration:
                    self.x = 0
                    self.trial_num += 1
                    self._iti_counter = None
            else:
                new_x = dx + self.x
                if new_x > self.track_length:
                    self.x = np.nan
                    self._iti_counter = 0
                else:
                    self.x = new_x

    def move(self, velocity, dt=1e-4):
        self._update_x(velocity, dt)
        return self.x

    def move_duration(self, duration, velocity, dt=1e-4):
        x = np.array([])
        for _ in range(int(duration / dt)):
            x = np.append(x, self.x)
            self._update_x(velocity, dt=dt)
        return x

    def move_n_trials(self, n_trials, velocity, dt=1e-4):
        x, trial_num = [], []
        while self.trial_num < n_trials:
            x.append(copy.copy(self.x))
            trial_num.append(copy.copy(self.trial_num))

            self._update_x(velocity, dt=dt)

        return np.array(x), np.array(trial_num)

    def reset(self):
        self.x = 0
        self.trial_num = 0
        self._iti_counter = None


class CircularMouse(SimMouse):

    pass


class OpenFieldMouse(SimMouse):

    pass

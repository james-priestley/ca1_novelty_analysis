from abc import abstractmethod

import pandas as pd

from lab3.core import Automorphism


class DecompositionStrategy(Automorphism):

    """Base class for signal decomposition strategies"""

    name = "Decomposition"

    def apply_to(self, experiment, channel='Ch2', label=None, from_type='dfof',
                 **kwargs):
        """Apply decomposition analysis to an experiment object.

        Parameters
        ----------
        experiment : lab3.experiment.base.ImagingExperiment
            Instance of ImagingExperiment to analyze
        channel : str or int, optional
        label : str, optional
        from_type : str {'dfof', 'raw'} or custom, optional

        Additional keyword arguments are passed directly to `calculate`.
        """
        sig_dict = self._load_signals(experiment, channel=channel, label=label,
                                      from_type='dfof')
        return self.calculate(**sig_dict, **kwargs)

    def calculate(self, signals):
        """Do analysis"""

        return self._decompose(signals)

    # ----------------- Implement these methods to subclass ----------------- #

    @abstractmethod
    def _decompose(self, signals):
        """Implement the decomposition algorithm here. It should return a
        dataframe the same width as signals, preserving the column index
        """

    # ------------------------------- Optional ------------------------------ #

    def _load_signals(self, experiment, channel, label, from_type):
        """By default only a single signal dataframe is passed to `calculate`
        when using `apply`. If you override `calculate` to take additional
        arguments in your custom strategy, you should load and add them to the
        dictionary here, which will be unpacked as arguments to `calculate`.
        """
        sig_dict = {
            'signals': experiment.signals(signal_type=from_type,
                                          channel=channel, label=label,
                                          max_frame=None)
        }
        return sig_dict


class LinearDynamicalSystem(DecompositionStrategy):

    """Fit a linear dynamical system using SSM.

    Parameters
    ----------
    n_components : int, optional
        Dimensionality of the latent space. Defaults to 1

    Additional keyword arguments are passed directly to ssm.LDS.fit during
    calculation.
    """

    def __init__(self, n_components=1, **model_kws):

        self.n_components = 1
        self.model_kws = model_kws

    def _decompose(self, signals):

        from ssm import LDS
        self.model = LDS(signals.shape[0], self.n_components, **self.model_kws)
        self.model_outputs = self.model.fit(signals.values.T)
        latents = pd.DataFrame(
            self.model_outputs[1].mean_continuous_states[0].T)
        latents.columns = signals.columns
        latents.index.name = 'component'

        return latents

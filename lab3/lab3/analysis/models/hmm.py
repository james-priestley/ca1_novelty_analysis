import sys

import numpy as np
import scipy.stats as sp
from scipy.special import factorial
from sklearn.cluster import KMeans
from sklearn.utils import check_random_state

try:
    from hmmlearn.base import _BaseHMM, ConvergenceMonitor
    from hmmlearn.utils import iter_from_X_lengths

except ImportError:
    raise ImportError("hmmlearn is not installed")

np.seterr(divide='ignore')


class RelativeConvergenceMonitor(ConvergenceMonitor):

    """A modified convergence monitor with more relaxed stopping criteria.
    Rather than evaluate the tolerance on the difference of likelihoods at
    iterations t and t_-1, we use the percentage change, i.e. the ratio of
    (t - t_-1) / t_-1."""

    # what do we need to override?

    def report(self, logprob):
        """Reports convergence to :data:`sys.stderr`.
        The output consists of three columns: iteration number, log
        probability of the data at the current iteration and convergence
        rate.  At the first iteration convergence rate is unknown and
        is thus denoted by NaN.

        Parameters
        ----------
        logprob : float
            The log probability of the data as computed by EM algorithm
            in the current iteration.
        """
        if self.verbose:
            delta = logprob - self.history[-1] if self.history else np.nan
            relative_delta = np.abs(delta / self.history[-1]) \
                if self.history else np.nan

            message = self._template.format(
                iter=self.iter + 1, logprob=logprob, delta=relative_delta)
            print(message, file=sys.stderr)

        self.history.append(logprob)
        self.iter += 1

    @property
    def converged(self):
        """``True`` if the EM algorithm converged and ``False`` otherwise."""
        if self.iter == self.n_iter:
            return True
        else:
            if len(self.history) == 2:
                delta = self.history[1] - self.history[0]
                relative_delta = np.abs(delta / self.history[1])
                return relative_delta < self.tol
            else:
                return False


class BaseMixin:

    """Mixin to fix issue in hmmlearn's _BaseHMM, which uses a deprecated
    private methods from sklearn's BaseEstimator class."""

    def _init(self, X, lengths):
        """Initializes model parameters prior to fitting.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Feature matrix of individual samples.
        lengths : array-like of integers, shape (n_sequences, )
            Lengths of the individual sequences in ``X``. The sum of
            these should be ``n_samples``.
        """
        init = 1. / self.n_components
        if 's' in self.init_params or not hasattr(self, "startprob_"):
            self.startprob_ = np.full(self.n_components, init)
        if 't' in self.init_params or not hasattr(self, "transmat_"):
            self.transmat_ = np.full((self.n_components, self.n_components),
                                     init)


class ScoringMixin:

    """Mixin class that provides some auxilary methods for scoring the models.
    Useful for doing model selection"""

    def bic(self, X, lengths=None):
        """Bayesian information criterion for the current model on the input X.

        Parameters
        ----------
        X : array of shape (n_samples, n_features)

        lengths : array of shape (n_trials,)
            Note sum(lengths)==n_samples

        Returns
        -------
        bic : float
            The lower the better.
        """
        return -2 * self.score(X, lengths=lengths) \
            + (self.n_components * (self.n_components - 1)
               + self.n_components * self.n_features) * np.log(X.shape[0])

    def aic(self, X, lengths=None):
        """Akaike information criterion for the current model on the input X.

        Parameters
        ----------
        X : array of shape (n_samples, n_features)

        lengths : array of shape (n_trials,)
            Note sum(lengths)==n_samples

        Returns
        -------
        aic : float
            The lower the better.
        """
        return -2 * self.score(X, lengths=lengths) \
            + (self.n_components * (self.n_components - 1)
               + self.n_components * self.n_features)

    def estimate_transitions(self, X, lengths=None, method='empirical',
                             algorithm='viterbi'):
        """Empirically estimate the transition matrix from the observation
        sequence(s). State transitions are tabulated along the Viterbi path
        by default.

        Parameters
        ----------
        X : array of shape (n_samples, n_features)

        lengths : array of shape (n_sequences), optional
            Length of each observation sequence in X.
            Note sum(lengths)==n_samples

        algorithm : str, optional
            Either "viterbi" or "map"

        Returns
        -------
        trans_mat_list : array of shape
                         (n_sequences, n_components, n_components)
            Empirical transition matrix for each observation sequence.

        # TODO could do this analytically as well
        """

        if lengths is None:
            lengths = [X.shape[0]]

        state_path = self.decode(X, lengths=lengths, algorithm=algorithm)[1]

        # cum_length = 0
        trans_mat_list = []
        # for length in lengths:
        for i, j in iter_from_X_lengths(X, lengths):
            if method == 'empirical':
                path_seg = state_path[i:j]

                trans_mat = np.zeros((self.n_components, self.n_components))
                for t in range(len(path_seg[:-1])):
                    trans_mat[path_seg[t], path_seg[t+1]] += 1

                trans_mat_list += [(trans_mat.T / np.sum(trans_mat, axis=1)).T]
            elif method == 'analytical':
                # TODO
                # Compute the m-step estimates of the transitions for each
                # trial
                raise NotImplementedError

        return np.stack(trans_mat_list)

    def estimate_emissions(self, X, lengths=None, dt=0.125,
                           method='empirical', algorithm='viterbi'):
        """Estimate the emission matrix from the observation sequence(s).
        Emissions can be estimated analytically from the Baum-Welch m-step,
        or empirically from the inferred state sequence.

        Parameters
        ----------
        X : array of shape (n_samples, n_features)

        lengths : array of shape (n_sequences), optional
            Length of each observation sequence in X.
            Note sum(lengths)==n_samples

        dt : float, optional
            Length of sample bin in X in seconds, default is 0.125. Estimated
            emissions are converted to firing rates via emissions * 1/dt.

        method : str, optional
            'empirical' (default) or 'analytical'

        algorithm : str, optional
            'viterbi' (default) or 'map'

        Returns
        -------
        emissions_list : array of shape(n_sequences, n_components, n_features)
            Estimated emissions matrix for each observation sequence
        """

        # emissions_list = []
        # if method=='analytical':
        #     posterior = self.predict_proba(X, lengths=lengths)
        #
        #     emissions_list = []
        #     for i, j in iter_from_X_lengths(X, lengths):
        #         num = np.dot(X[i:j].T, posterior[i:j])
        #         den = np.sum(posterior[i:j], axis=0)
        #
        #         emissions_list += [-1 * (1/dt) * np.log(1 - (num/den))]
        #
        # elif method=='empirical':
        #     for i, j in iter_from_X_lengths(X, lengths):
        #         state_path = self.decode(X[i:j], algorithm=algorithm)[1]
        #
        #         # for each neuron, calculate the probability it spikes in
        #         # each state...
        #         emissions_list += [1/dt * np.stack([
        #             np.mean(X[i:j][state_path==m, :], axis=0) \
        #             for m in range(self.n_components)])]
        #
        # return np.nan_to_num(np.stack(emissions_list))

        # TODO this will depend on the emissions model; how to generalize?

        raise NotImplementedError

    def _compute_shuffle_posterior(self, X, lengths, trial_type):
        # """Doc string"""
        # np.random.seed()
        #
        # shuffle_trials = lambda m: np.dstack(
        #     [m[np.random.permutation(m.shape[0]), :, x] \
        #          for x in range(m.shape[-1])])
        #
        # sX = X.copy().reshape(len(lengths), -1, X.shape[-1])
        # sX = shuffle_trials(sX).reshape(-1, sX.shape[-1])
        #
        # return self.predict_proba(sX, lengths=lengths)

        # TODO this will depend on the emissions model; how to generalize?

        raise NotImplementedError

    def calc_posterior_ci(self, X, lengths, percentile=95, n_shuffle=10,
                          n_processes=1):
        """Calculate null confidence intervals on the posterior probabilities
        of the hidden states. This is done by shuffling the trials for each
        neuron independently to destroy neuron-neuron correlations, while
        leaving their average firing rate statistics intact. Results are stored
        as an attribute of the model, as:
            self.posterior_ci_

        Parameters
        ----------
        X : array of shape (n_samples, n_features)

        lengths : array of shape (n_trials,)
            Note sum(lengths)==n_samples

        percentile : float, optional
            Percentile of the null distribution to store, default 95.

        n_shuffle : int, optional
            Number of shuffle datasets to generate, default 10.

        n_processes : int, optional
            Spawn a parallel pool with multiple workers, default 1.
        """
        # p = Pool(n_processes) if n_processes > 1 else None
        #
        # if p:
        #     shuffle_post = p.map(lambda x: self._compute_shuffle_posterior(
        #         X, lengths), np.arange(n_shuffle))
        # else:
        #     shuffle_post = map(lambda x: self._compute_shuffle_posterior(
        #         X, lengths), np.arange(n_shuffle))
        #
        # all_shuffles = np.concatenate(shuffle_post, axis=0)
        # self.posterior_ci_ = [np.nanpercentile(all_shuffles[:, x], percentile)
        #                       for x in range(all_shuffles.shape[1])]

        # TODO this will depend on the emissions model; how to generalize?
        raise NotImplementedError


class PoissonHMM(BaseMixin, _BaseHMM, ScoringMixin):

    """Inference for Hidden Markov models with independent Poisson emissions.

    Parameters
    ----------
    n_components : int
        Number of states.
    startprob_prior : array, shape (n_components, )
        Initial state occupation prior distribution. Optional.
    transmat_prior : array, shape (n_components, n_components)
        Matrix of prior transition probabilities between states. Optional.
    means_prior : float
        Optional.
    algorithm : string
        Decoder algorithm. Must be one of "viterbi" or "map".
        Defaults to "viterbi".
    random_state : RandomState or an int seed
        A random number generator instance.
    n_iter : int, optional
        Maximum number of iterations to perform.
    tol : float, optional
        Convergence threshold. EM will stop if the gain in log-likelihood
        relative to initial log-likelihood is below this value.
    verbose : bool, optional
        When ``True`` per-iteration convergence reports are printed
        to :data:`sys.stderr`. You can diagnose convergence via the
        :attr:`monitor_` attribute.
    params : string, optional
        Controls which parameters are updated in the training process.  Can
        contain any combination of 's' for startprob, 't' for transmat, and
        'm' for means of the observation Poisson distributions. Defaults to all
        parameters.
    init_params : string, optional
        Controls which parameters are initialized prior to training. See params
        for options and defaults.
    k_means_init : bool, optional
        When ``True`` emission matrix means are initialized by a modified
        k-means clustering of the observations.
    combine_obs_seq :
        NotImplemented

    Attributes
    ----------
    monitor_ : ConvergenceMonitor
        Monitor object used to check the convergence of EM.
    startprob_ : array, shape (n_components, )
        Initial state occupation distribution.
    transmat_ : array, shape (n_components, n_components)
        Matrix of transition probabilities between states.
    means_ : array, shape (n_components, n_features)
        Matrix of means for the observations Poisson distributions.
    """

    def __init__(self, n_components=1, startprob_prior=1.0, transmat_prior=1.0,
                 means_prior=0.0001, algorithm="viterbi", random_state=None,
                 n_iter=100, tol=1e-5, verbose=False, params="stm",
                 init_params="stm", kmeans_init=True):
        _BaseHMM.__init__(self, n_components, startprob_prior=startprob_prior,
                          transmat_prior=transmat_prior, algorithm=algorithm,
                          random_state=random_state, n_iter=n_iter, tol=tol,
                          params=params, verbose=verbose,
                          init_params=init_params)
        self.kmeans_init = kmeans_init
        self.means_prior = means_prior

    def _check(self):
        super()._check()

        self.means_ = np.asarray(self.means_)
        self.n_features = self.means_.shape[1]

    def _init(self, X, lengths=None):
        super()._init(X, lengths=lengths)
        self.random_state = check_random_state(self.random_state)

        _, n_features = X.shape
        if hasattr(self, 'n_features') and self.n_features != n_features:
            raise ValueError('Unexpected number of dimensions, got %s but '
                             'expected %s' % (n_features, self.n_features))

        self.n_features = n_features
        if 'm' in self.init_params or not hasattr(self, "means_"):
            if self.kmeans_init:
                kmeans = KMeans(n_clusters=self.n_components,
                                random_state=self.random_state)
                kmeans.fit(X)
                self.means_ = kmeans.cluster_centers_
                self.means_ += 0.1 * self.random_state.rand(
                    self.n_components, self.n_features)
                self.means_[self.means_ < self.means_prior] = self.means_prior

            else:
                self.means_ = self.random_state.rand(self.n_components,
                                                     self.n_features)

    def _compute_log_likelihood(self, X):
        # we should store the factorial of X so that we don't have to keep
        # recomputing it
        term1 = -1 * np.ones((X.shape[0], self.n_components)) * \
                            np.sum(self.means_, axis=1)
        term2 = np.dot(X, np.log(self.means_.T))
        term3 = np.sum(np.log(factorial(X)), axis=1).reshape(-1, 1)
        return term1 + term2 - term3

    def _initialize_sufficient_statistics(self):
        stats = super()._initialize_sufficient_statistics()
        stats['post'] = np.zeros(self.n_components)
        stats['obs'] = np.zeros((self.n_components, self.n_features))
        return stats

    def _accumulate_sufficient_statistics(self, stats, X, framelogprob,
                                          posteriors, fwdlattice, bwdlattice):
        super()._accumulate_sufficient_statistics(
            stats, X, framelogprob, posteriors, fwdlattice, bwdlattice)

        if 'm' in self.params:
            stats['obs'] += np.dot(posteriors.T, X)
            stats['post'] += np.sum(posteriors, axis=0)

    def _do_mstep(self, stats):
        super()._do_mstep(stats)

        denom = stats['post'][:, np.newaxis]

        if 'm' in self.params:
            self.means_ = stats['obs'] / denom

            # regularize
            self.means_[self.means_ < self.means_prior] = self.means_prior

    def _generate_sample_from_state(self, state, random_state=None):
        raise NotImplementedError
        # random_state = check_random_state(random_state)
        # return [(self.emissionprob_[state, :] > random_state.rand()).argmax()]


class BernoulliHMM(BaseMixin, _BaseHMM, ScoringMixin):

    """Inference for Hidden Markov models with independent Bernoulli emissions.

    Parameters
    ----------
    n_components : int
        Number of states.
    startprob_prior : array, shape (n_components, )
        Initial state occupation prior distribution.
    transmat_prior : array, shape (n_components, n_components)
        Matrix of prior transition probabilities between states.
    algorithm : string
        Decoder algorithm. Must be one of "viterbi" or "map".
        Defaults to "viterbi".
    random_state: RandomState or an int seed
        A random number generator instance.
    n_iter : int, optional
        Maximum number of iterations to perform.
    tol : float, optional
        Convergence threshold. EM will stop if the gain in log-likelihood
        relative to initial log-likelihood is below this value.
    verbose : bool, optional
        When ``True`` per-iteration convergence reports are printed
        to :data:`sys.stderr`. You can diagnose convergence via the
        :attr:`monitor_` attribute.
    params : string, optional
        Controls which parameters are updated in the training
        process.  Can contain any combination of 's' for startprob,
        't' for transmat, and other characters for subclass-specific
        emission parameters. Defaults to all parameters.
    init_params : string, optional
        Controls which parameters are initialized prior to
        training.  Can contain any combination of 's' for
        startprob, 't' for transmat, and other characters for
        subclass-specific emission parameters. Defaults to all
        parameters.
    k_means_init : bool, optional
        When ``True`` emission matrix probabilities are initialized by a
        modified k-means clustering of the observations.

    Attributes
    ----------
    monitor_ : ConvergenceMonitor
        Monitor object used to check the convergence of EM.
    startprob_ : array, shape (n_components, )
        Initial state occupation distribution.
    transmat_ : array, shape (n_components, n_components)
        Matrix of transition probabilities between states.
    emissionprob_ : array, shape (n_components, n_features)
    """

    def __init__(self, n_components=1, startprob_prior=1.0, transmat_prior=1.0,
                 emission_prior=0.0065, algorithm="viterbi", random_state=None,
                 n_iter=100, tol=1e-5, verbose=False, params="ste",
                 init_params="ste", kmeans_init=True,
                 relative_convergence=True):
        _BaseHMM.__init__(self, n_components, startprob_prior=startprob_prior,
                          transmat_prior=transmat_prior, algorithm=algorithm,
                          random_state=random_state, n_iter=n_iter, tol=tol,
                          params=params, verbose=verbose,
                          init_params=init_params)
        self.kmeans_init = kmeans_init
        self.emission_prior = emission_prior
        if relative_convergence:
            self.monitor_ = RelativeConvergenceMonitor(self.tol, self.n_iter,
                                                       self.verbose)

    def _check(self):
        super()._check()

        self.emissionprob_ = np.asarray(self.emissionprob_)
        self.n_features = self.emissionprob_.shape[1]

    def _init(self, X, lengths=None):
        super()._init(X, lengths=lengths)
        self.random_state = check_random_state(self.random_state)

        _, n_features = X.shape
        if hasattr(self, 'n_features') and self.n_features != n_features:
            raise ValueError('Unexpected number of dimensions, got %s but '
                             'expected %s' % (n_features, self.n_features))

        self.n_features = n_features
        if 'e' in self.init_params or not hasattr(self, "emissionprob_"):
            if self.kmeans_init:
                kmeans = KMeans(n_clusters=self.n_components,
                                random_state=self.random_state)
                kmeans.fit(np.nan_to_num(sp.zscore(X)))
                self.emissionprob_ = kmeans.cluster_centers_
                self.emissionprob_[self.emissionprob_ < 0] = 0
                self.emissionprob_ /= (np.max(self.emissionprob_) * 1.25)
                self.emissionprob_ += 0.1 * self.random_state.rand(
                    self.n_components, self.n_features)
                self.emissionprob_ /= (np.max(self.emissionprob_) * 1.25)
            else:
                self.emissionprob_ = self.random_state.rand(self.n_components,
                                                            self.n_features)

    def _compute_log_likelihood(self, X):
        term1 = np.nan_to_num(np.dot(X == 1, np.log(self.emissionprob_.T)))
        term2 = np.nan_to_num(np.dot(X == 0, np.log(1 - self.emissionprob_.T)))
        return term1 + term2

    def _generate_sample_from_state(self, state, random_state=None):
        random_state = check_random_state(random_state)
        return [(self.emissionprob_[state, :] > random_state.rand()).argmax()]

    def _initialize_sufficient_statistics(self):
        stats = super()._initialize_sufficient_statistics()
        stats['obs'] = np.zeros((self.n_components, self.n_features))
        stats['post'] = np.zeros((self.n_components))
        return stats

    def _accumulate_sufficient_statistics(self, stats, X, framelogprob,
                                          posteriors, fwdlattice, bwdlattice):
        super()._accumulate_sufficient_statistics(
            stats, X, framelogprob, posteriors, fwdlattice, bwdlattice)

        if 'e' in self.params:
            stats['obs'] += np.dot(posteriors.T, X)
            stats['post'] += np.sum(posteriors, axis=0)

    def _do_mstep(self, stats):
        super()._do_mstep(stats)

        if 'e' in self.params:
            denom = stats['post'][:, np.newaxis]
            self.emissionprob_ = stats['obs'] / denom

            # regularize
            self.emissionprob_[self.emissionprob_ < self.emission_prior] = \
                self.emission_prior

    def set_params(**params):
        """Set the model parameters a priori"""
        # will be useful for storing model information
        raise NotImplementedError

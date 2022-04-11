import numpy as np
#import scipy.stats as sp
import pandas as pd
from sklearn.decomposition import NMF
from sklearn.cluster import KMeans


class NormalizedNMF(NMF):

    """Identical to sklearn's NMF, but the components are scaled so that the
    maximum of each component's pattern is 1. If the components' patterns are
    interpreted as per-lap place fields, this allows the component weights to
    be more directly interpreted as per-lap place field amplitudes.
    """

    def fit_transform(self, *args, **kwargs):
        Y = super().fit_transform(*args, **kwargs)
        self.components_ *= Y.max(axis=0).reshape(-1, 1)
        return self.transform(*args, **kwargs)

    def sort_components(self, X, n_laps, reorder_first_lap=True):
        """Re-order NMF components according to the peak of their
        spatiotemporal pattern. This method returns the sorted, transformed
        signals, but also re-orders the components_ attribute of the estimator.

        Parameters
        ----------
        X : array of shape (n_samples, n_features)
        n_laps : int
            Number of laps. Used for reshaping transformed signals Y into
            an array of shape (n_laps, n_spatial_bins, n_components).
        reorder_first_lap : bool, optional
            If True, inspect the first two sorted components (i.e. the two that
            come earliest in space x trials). Re-index components so the later
            of those two components is now the last component in the sorted
            array. Useful for moving the `plateau' component to the end of the
            array.
        """
        Y = self.transform(X)
        component_rasters = Y.reshape(n_laps, -1, self.n_components)
        lap_means = component_rasters.mean(axis=1)

        sort_idx = np.argsort(np.argmax(lap_means, axis=0))

        if reorder_first_lap:
            first_lap_order = np.argsort(np.argmax(
                component_rasters[0, :, sort_idx[:2]], axis=-1))
            sort_idx = np.concatenate(
                [[sort_idx[first_lap_order[0]]],
                 sort_idx[2:],
                 [sort_idx[first_lap_order[1]]]]
            )

        self.components_ = self.components_[sort_idx]
        return Y[:, sort_idx]


def preprocess_fields(field_info):
    """Flatten all rasters and normalize by their mean firing rate.

    Parameters
    ----------
    field_info : pd.DataFrame
        From ContextSwitchDataset.nmf_field_info

    Returns
    -------
    X : array, (n_)
    """

    X = np.stack([r.flatten() for r in field_info['raster']]).T
    X /= X.mean(axis=0)

    return X


def model_selection(X, n_min=2, n_max=20):
    """doc string"""

    mdls = []
    err = []
    for n in range(n_min, n_max + 1):
        print(n, end='\r')
        mdl = NormalizedNMF(n_components=n, l1_ratio=1, alpha=0,
                            max_iter=10000, tol=1e-4, solver='cd',
                            init='nndsvd', verbose=False)
        mdl.fit(X)

        mdls.append(mdl)
        err.append(mdl.reconstruction_err_)

    return mdls, err


class BTSPCluster(KMeans):

    def _preprocess_components(self, nmf_components, fit=False):
        """Note the z-scored components will be transposed to fit sklearn
        standard"""
        if fit:
            self._cmean = nmf_components.mean(axis=1)
            self._cstd = nmf_components.std(axis=1)

        return ((nmf_components.T - self._cmean) / self._cstd)

    def _find_true_centers(self, nmf_components, labels):
        centers = np.stack([nmf_components[:, labels == i].mean(axis=1)
                            for i in np.unique(labels)])
        centers = np.hstack([centers[:, 0].reshape(self.n_clusters, -1),
                             centers[:, -1].reshape(self.n_clusters, -1),
                             centers[:, 1:-1]])
        centers[self.btsp_label_] *= -1
        centers = pd.DataFrame(centers, index=self.cluster_names).melt(
                ignore_index=False, var_name='pattern', value_name='weight'
            ).reset_index()
        return centers

    def fit(self, nmf_components, **kwargs):
        """
        Parameters
        ----------
        nmf_components : array of shape (n_components, n_place_fields)
            Note this is transposed from the regular KMeans input, mainly for
            the convenience of passive the NMF object's attribute directly.
            We also z-score each component's weights before fitting.

        Additional keyword arguments are passed directly to KMeans.fit
        """

        z_components = self._preprocess_components(nmf_components, fit=True)
        super().fit(z_components, **kwargs)

        self.btsp_label_ = np.argmax(self.cluster_centers_[:, -1])
        self.true_cluster_centers_ = self._find_true_centers(
            nmf_components, self.labels_)

        return self

    @property
    def cluster_names(self):
        num_patterns = self.cluster_centers_.shape[0]
        names = []
        other_count = 0
        for n in range(num_patterns):
            if n == self.btsp_label_:
                names.append('btsp')
            else:
                names.append(f'other_{other_count}')
                other_count += 1
        return names

    def predict(self, nmf_components, **kwargs):
        return super().predict(
            self._preprocess_components(nmf_components), **kwargs)

    def fit_predict_btsp(self, nmf_components, **kwargs):
        """Like fit_predict, but returns a boolean whether each sample is in
        the BTSP cluster or not"""
        labels = super().fit_predict(nmf_components, **kwargs)
        return labels == self.btsp_label_

    @property
    def true_cluster_centers(self):
        return self.true_cluster_centers_

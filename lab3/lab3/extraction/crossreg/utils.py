import os

import numpy as np
import pandas as pd
import cv2
from scipy.sparse import csc_matrix
from scipy.optimize import linear_sum_assignment
from skimage.measure import find_contours

from lab3.misc.progressbar import ProgressBar

CV2_CRITERIA = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 1000, 1e-10)


def get_suite2p_time_average(ds):
    """
    Parameters
    ----------
    ds: sima.ImagingDataset

    Returns
    -------
    array (num_planes, num_rows, num_columns)
    """
    ops_list = np.load(os.path.join(ds.savedir, "suite2p", "ops1.npy"),
                       allow_pickle=True)
    return np.stack([ops['meanImgE'] for ops in ops_list])


def crop_image(image, crop):
    """Maybe write a function to do multi-z crops? If only one crop tuple is
    passed, then use it for all planes?"""
    pass


def find_crop_coords(image, threshold=1e-10):
    """Find the columns and rows of an image that are of constant value.

    Parameters
    ----------
    image : array (n_rows, n_columns)
    threshold : float, optional
        Threshold on the variance of values in a row or column, for determining
        whether the data is constant.

    Returns
    -------
    rows, columns : array
        Lists of rows and columns to crop
    """

    rows = image.var(axis=1) > threshold
    columns = image.var(axis=0) > threshold

    return rows, columns


def float_to_uint(image):
    image -= np.min(image)
    image /= np.max(image)
    image *= 255
    return image.astype('uint8')


def compute_transforms(images, reference_idx=0, from_reference=False):
    """Compute an affine transform between each image and a chosen reference
    image.

    Parameters
    ----------
    images : list
    reference_idx : int, optional
        List index of the reference image. Defaults to 0.

    Returns
    -------
    warp_list : list
        List of warp matrices for computed transforms.

    """
    # TODO - add multiprocessing?

    p = ProgressBar(len(images))
    warp_list = []
    for i, current_image in enumerate(images):
        p.update(i)
        warp_list.append(affine_registration(images[reference_idx],
                                             current_image))
    p.update(len(images))
    print(" Done!")
    return warp_list


def affine_registration(image0, image1):
    """Two step registration. First estimate a rigid translation between the
    two images, and then use this to seed the affine transformation
    optimization

    Parameters
    ----------
    image0, image1 : array
        TODO : Which is target, which is reference?
    """

    warp_matrix = np.eye(2, 3, dtype=np.float32)

    image1 = float_to_uint(image1)
    image0 = float_to_uint(image0)

    # First Translate converted raw images
    (_, warp_matrix) = cv2.findTransformECC(image1, image0, warp_matrix,
                                            cv2.MOTION_TRANSLATION,
                                            CV2_CRITERIA, None, 1)

    # Then seed affine transform for processed images
    (_, warp_matrix) = cv2.findTransformECC(image1, image0, warp_matrix,
                                            cv2.MOTION_AFFINE, CV2_CRITERIA,
                                            None, 1)

    return warp_matrix


def calc_distance_matrix(masks0, masks1):
    """Calculates a matrix of Jaccard distances between the columns of the
    binary input matrices, equal to 1 minus the intersection / union. Uses
    sparse matrices to conserve memory when dealing with a large number of
    ROI pairs."""

    flat_masks = np.vstack([np.stack([m.flatten() for m in masks])
                            for masks in [masks0, masks1]])
    mat = csc_matrix(flat_masks.T)

    cols_sum = mat.getnnz(axis=0)
    ab = mat.T * mat
    aa = np.repeat(cols_sum, ab.getnnz(axis=0))
    bb = cols_sum[ab.indices]

    similarities = ab.copy()
    similarities.data = similarities.data / (aa + bb - ab.data)

    return 1 - similarities.todense()[0:len(masks0), len(masks0):]


def binarize_mask(roi, thres=0.05):
    """Return binarized numpy array given an ROI object"""
    mask = np.sum(roi.__array__(), axis=0)
    return mask > (np.max(mask) * thres)


def calc_mask_centroid(mask):
    return [np.mean(l) for l in np.where(mask)]


def calc_mask_contour(mask, level=.99):
    if np.sum(mask) == 0:
        return None
    else:
        return find_contours(mask, level=level)[0].T[::-1]


def apply_transform(image, transform, sz, invert=False):
    if invert:
        transform = cv2.invertAffineTransform(transform)
    return cv2.warpAffine(image.astype(np.uint8), transform, (sz[1], sz[0]),
                          flags=cv2.INTER_LINEAR).astype(np.int)


def get_roi_info(roi, crop=None, transform=None, target_shape=None):
    """Returns some some basic information about an ROI object

    Parameters
    ----------
    roi : sima.ROI
    crop : list of list of arrays
    transform : list of arrays, optional
        Affine transform to apply, per plane
    target_shape : tuple, optional
        (num_planes, num rows, num_columns)

    Returns
    -------
    pd.Series containing fields:
        roi: sima.ROI
        plane: int
        mask: 2D array
        contour: list of arrays

    """
    mask = binarize_mask(roi)
    plane = int(roi.label[3])
    if crop is not None:
        rows, columns = crop[plane]
        mask = mask[rows][:, columns]

    if transform is not None:
        mask = apply_transform(mask, transform[plane], target_shape[1:])

    return pd.Series({'roi': roi,
                      'plane': int(roi.label[3]),
                      'mask': mask,
                      'centroid': calc_mask_centroid(mask),
                      'contour': calc_mask_contour(mask)},
                     name=roi.label)


def find_nearby_rois(centroid, reference_rois, max_distance=25,
                     exclude_partnered=None):
    """Filter the reference ROIs for ones whose centroids are nearby the
    current ROI.

    Parameters
    ----------
    centroid : list
    reference_rois : pd.DataFrame
    max_distance : float, optional
        Maximum centroid distance between the target and reference. Defaults to
        25 pixels
    exclude_partnered : str, optional
        savedir of an ImagingDataset. Exclude any reference ROIs that are
        already partnered to an ROI from this dataset.

    Returns
    -------
    eligible_references : pd.DataFrame
    """
    eligible_references = reference_rois[[exclude_partnered not in partners for
                                          partners in reference_rois.partners]]
    distances = np.abs(np.stack(eligible_references.centroid)
                       - np.asarray(centroid))
    return eligible_references[(distances < max_distance).all(axis=1)]

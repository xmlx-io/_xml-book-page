"""
XML Book Surrogate Module
=========================

This module implements a collection of surrogate explainer functions used by
the book.
"""

# Author: Kacper Sokol <kacper@xmlx.io>
#         Alex Hepburn <ah13558@bristol.ac.uk>
# License: MIT

import scipy
import scipy.stats

import numpy as np

__all__ = ['gini_index', 'entropy', 'mse', 'get_hyperrectangle_indices',
           'weighted_purity', 'one_hot_encode', 'get_bin_sampling_values',
           'undiscretise_data']


def gini_index(x):
    """
    Computes a Gini Index of a numpy array.

    Parameters
    ----------
    x : 1-dimensional numpy array
        An array with class predictions or labels.

    Returns
    -------
    gini : float
        Gini Index of the ``x`` array.
    """
    x_ = np.asarray(x)
    _, counts = np.unique(x_, return_counts=True)
    frequencies = counts / x_.shape[0]

    gini_itemwise = frequencies * (1 - frequencies)

    gini = np.sum(gini_itemwise)

    assert 0 <= gini <= 1

    return gini


def entropy(x, base=None):
    """
    Computes entropy of a numpy array.

    Parameters
    ----------
    x : 1-dimensional numpy array
        An array with class predictions or labels.
    base : integer, optional (default=None)
        Base of the logarithm used for computing entropy. By default
        (``None``), the natural logarithm is used.

    Returns
    -------
    entropy_ : float
        Entropy of the ``x`` array.
    """
    assert base is None or isinstance(base, int), 'Wrong type.'

    _, counts = np.unique(x, return_counts=True)
    entropy_ = scipy.stats.entropy(counts, base=base)

    return entropy_


def mse(x):
    """
    Computes Mean Squared Error of a numpy array.

    Parameters
    ----------
    x : 1-dimensional numpy array
        An array with regression or probabilistic predictions.

    Returns
    -------
    mse_ : float
        Mean Squared Error of the ``x`` array.
    """
    # Error
    err = x - np.mean(x)
    # Squared error
    err_sq = np.square(err)
    # Mean Squared Error
    mse_ = np.mean(err_sq)

    return mse_


def get_hyperrectangle_indices(discretised_data, hyperrectangle):
    """
    Extracts row indices of a data array that match the specified sample.

    This function returns row indices of the ``discretised_data`` array that
    are identical to the ``hyperrectangle`` array.
    The data set has to be discretised, i.e., all of its values have to be
    between 0 and 3 inclusive.

    Parameters
    ----------
    discretised_data : 2-dimensional numpy array
        A 2-dimensional array with data.
    hyperrectangle : 1-dimensional numpy array
        A 1-dimensional array that will be matched against each row of the
        ``discretised_data`` array.

    Returns
    -------
    matching_indices : 1-dimensional numpy array
        An array with indices of the matching rows.
    """
    import fatf.utils.array.validation as fatf_v
    hyperrectangle_ = np.asarray(hyperrectangle)

    assert np.all(0 <= discretised_data), 'Data probably not discretised.'
    if not fatf_v.is_1d_array(discretised_data):
        assert (discretised_data.shape[1]
                == hyperrectangle_.shape[0]), 'Size mismatch.'

    matching_rows = (discretised_data == hyperrectangle)
    if not fatf_v.is_1d_array(discretised_data):
        matching_rows = matching_rows.all(axis=1)
    matching_indices = np.where(matching_rows)[0]

    return matching_indices


def weighted_purity(discretised_data, labels, metric):
    """
    Computes weighted purity metric of ``labels`` based on grouping given by
    unique encodings in the ``discretised_data`` array.

    This function identifies unique rows in the ``discretised_data`` array and
    computes user-selected purity metric for the corresponding labels
    (extracted from the ``labels`` array).
    The final result is computed as a weighted average of these individual
    metrics, where the weights are proportions of instances used to compute
    each individual metric.

    The data set (``discretised_data``) has to be discretised, i.e., all of its
    values have to be between 0 and 3 inclusive.
    The labels (``labels``) are either:

    * *crisp* predictions of a classifier or *class* labels; or
    * *numbers* representing regression values or *probabilistic* predictions
      for a single class (in case of probabilistic classifiers).

    The ``metric`` has to be chosen appropriately to the type of ``labels``:

    * ``'mse'`` (Mean Squared Error) used for **numerical** ``labels``; and
    * ``'gini'`` (Gini Index) used for **crisp** ``labels``.

    Parameters
    ----------
    discretised_data : 2-dimensional numpy array
        A 2-dimensional array with *discretised* data.
    labels : 1-dimensional numpy array
        A 1-dimensional array with labels either holding *numbers* (regression
        values or probabilistic predictions of a single class) or *crisp*
        labels (class predictions or ground truth labels).
    metric : string
        Either ``'mse'`` for Mean Squared Error or ``'gini'`` for Gini Index.

    Returns
    -------
    weighted_purity_ : float
        A weighted purity (``metric``) of the ``labels`` based on the partition
        of the ``discretised_data`` array.
    """
    import fatf.utils.array.validation as fatf_v
    assert np.all(0 <= discretised_data), 'Data probably not discretised.'
    #
    assert (discretised_data.shape[0] == labels.shape[0]), 'Size mismatch.'
    #
    assert metric.lower() in ('mse', 'gini'), (
        'Incorrect metric specifier. Should either be *mse* or *gini*.')

    items_count = discretised_data.shape[0]
    unique_hr = np.unique(discretised_data, axis=0)

    metric_fn = mse if metric.lower() == 'mse' else gini_index

    individual_purity = []
    for encoding in unique_hr:
        matching_rows = (discretised_data == encoding)
        if not fatf_v.is_1d_array(discretised_data):
            matching_rows = matching_rows.all(axis=1)
        matching_indices = np.where(matching_rows)[0]

        hr_metric = metric_fn(labels[matching_indices])
        hr_items_count = matching_indices.shape[0]

        individual_purity.append(hr_metric * hr_items_count)

    weighted_purity_ = np.sum(individual_purity) / items_count

    return weighted_purity_


def one_hot_encode(vector):
    """
    One-hot-encode the ``vector``.

    The order of one-hot-encoding is based on sorted unique values of the
    ``vector``.

    Parameters
    ----------
    vector : 1-dimensional numpy array
        A 1-dimensional array with *discrete* values.

    Returns
    -------
    ohe : 2-dimensional numpy array
        A binary 2-dimensional array with one-hot-encoded ``vector``.
    """
    import fatf.utils.array.validation as fatf_v

    vector = np.asarray(vector)
    assert fatf_v.is_1d_array(vector), 'vector has to be 1-D.'

    unique = np.sort(np.unique(vector))
    unique_count = unique.shape[0]

    ohe = np.zeros((vector.shape[0], unique_count), dtype=np.int8)

    for i, v in enumerate(unique):
        indices = get_hyperrectangle_indices(vector, v)
        ohe[indices, i] = 1

    return ohe


def get_bin_sampling_values(dataset, discretiser):
    """
    Captures the mean and standard deviation of the ``dataset`` for each
    hyper-rectangle encoded by the ``discretiser``.

    Parameters
    ----------
    dataset : 2-dimensional numpy array
        A data set to be analysed.
    discretiser : fat-forensics discretiser object
        A (fitted) discretiser that is compatible with the ``dataset``.

    Returns
    -------
    bin_sampling_values : dictionary of dictionaries holding 4-tuples
        The outer dictionary indicates the feature in the original data
        representation; the inner dictionary signifies a partition (quartile)
        of this feature. Under these two keys, a four-tuple holds the:
        minimum, maximum, mean and standard deviation (in this order)
        values of data points within this partition.
    """
    dataset_discretised = discretiser.discretise(dataset)

    bin_sampling_values = {}
    for index in range(discretiser.features_number):
        bin_sampling_values[index] = {}

        discretised_feature = dataset_discretised[:, index]
        feature = dataset[:, index]

        # The bin IDs need to be sorted as they are retrieved from
        # dictionary keys (hence may come in a random order), therefore
        # interfering with the enumerate procedure.
        bin_ids = sorted(
            list(discretiser.feature_value_names[index].keys()))
        bin_boundaries = discretiser.feature_bin_boundaries[index]
        for bin_i, bin_id in enumerate(bin_ids):
            bin_feature_indices = (discretised_feature == bin_id)
            bin_feature_values = feature[bin_feature_indices]

            # If there is data in the bin, get its empirical mean and
            # standard deviation, otherwise use numpy nan.
            # If there are no data in a bin, the frequency of this bin
            # will be 0, therefore data will never get sampled from this
            # bin, i.e., there will be no attempt to undiscretised them.
            if bin_feature_values.size:
                mean_val = bin_feature_values.mean()
                std_val = bin_feature_values.std()
            else:
                mean_val = np.nan
                std_val = np.nan

            # Use the true bin boundaries (extracted from the discretiser).
            # For the edge bins (with -inf and +inf edges) use the
            # empirical minimum and maximum (if possible) to avoid problems
            # with reverse sampling (see the _undiscretise_data method).
            if bin_i == 0:
                if bin_feature_values.size:
                    min_val = bin_feature_values.min()
                else:
                    min_val = -np.inf  # pragma: nocover
                    assert False, (  # pragma: nocover
                        'Since the upper bin boundary is inclusive in '
                        'the quartile discretiser this can never happen.')
                max_val = bin_boundaries[bin_i]
            # This is bin id count (+1) and not bind boundary count.
            elif bin_i == bin_boundaries.shape[0]:
                min_val = bin_boundaries[bin_i - 1]
                if bin_feature_values.size:
                    max_val = bin_feature_values.max()
                else:
                    max_val = np.inf
            else:
                min_val = bin_boundaries[bin_i - 1]
                max_val = bin_boundaries[bin_i]

            bin_sampling_values[index][bin_id] = (min_val, max_val,
                                                  mean_val, std_val)

    return bin_sampling_values


def undiscretise_data(discretised_data, discretiser, dataset):
    """
    Transforms discretised data back into their original representation.

    This function uses truncated normal sampling fitted into each
    hyper-rectangle.

    Parameters
    ----------
    discretised_data : 2-dimensional numpy array
        A discretised data set (in quartile representation) to be
        undiscretised.
    discretiser : fat-forensics discretiser object
        A (fitted) discretiser that is compatible with the ``dataset``
        (used to extract boundaries of hyper-rectangles).
    dataset : 2-dimensional numpy array
        A data set used to extract mean and standard deviation of each
        hyper-rectangle.

    Returns
    -------
    bin_sampling_values : 2-dimensional numpy array
        Undiscretised ``discretised_data``.
    """
    bin_sampling_values = get_bin_sampling_values(dataset, discretiser)
    dataset_dtype = dataset.dtype

    # Create a placeholder for undiscretised data. We copy the discretised
    # array instead of creating an empty one to preserve the values of
    # sampled categorical features, hence we do not need to copy them
    # later on. We also need to change the type of the array to correspond
    # to the original dataset.
    undiscretised_data = discretised_data.copy().astype(dataset_dtype)

    for index in range(discretised_data.shape[1]):
        discretised_column = discretised_data[:, index]
        undiscretised_column = undiscretised_data[:, index]

        unique_column_values = np.unique(discretised_column)
        for bin_id, bin_values in bin_sampling_values[index].items():
            if bin_id in unique_column_values:
                # Since sampling values must have been found in this bin,
                # there should be an empirical mean (2) and
                # standard deviation (3).

                bin_indices = np.where(discretised_column == bin_id)[0]
                samples_number = bin_indices.shape[0]

                min_, max_, mean_, std_ = bin_values
                if std_:
                    lower_bound = (min_ - mean_) / std_
                    upper_bound = (max_ - mean_) / std_

                    unsampled = scipy.stats.truncnorm.rvs(
                        lower_bound,
                        upper_bound,
                        loc=mean_,
                        scale=std_,
                        size=samples_number)
                else:
                    unsampled = np.array(samples_number * [mean_])

                undiscretised_column[bin_indices] = unsampled
    return undiscretised_data

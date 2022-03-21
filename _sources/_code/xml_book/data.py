"""
XML Book Data Module
====================

This module implements a collection of data functions used by the book.
"""

# Author: Kacper Sokol <kacper@xmlx.io>
#         Alex Hepburn <ah13558@bristol.ac.uk>
# License: MIT

import io
import requests
import zipfile

import sklearn.datasets
import sklearn.model_selection
import sklearn.preprocessing

import numpy as np

__all__ = ['generate_2d_moons', 'generate_bikes', 'get_boston']


def generate_2d_moons(random_seed=None):
    """
    Generates a two-dimensional *Two Moons* data set.

    The data set has 1200 training instances and 300 test instances.
    It is generated with 0.25 noise parameter and its features are scaled
    to the [0, 1] range.

    For reproducibility of the data sampling and train/test split,
    you may wish to set the ``random_seed`` parameter.

    Parameters
    ----------
    random_seed : integer, optional (default=None)
        A random seed used to initialise Python's and numpy's ``random``
        modules. If ``None``, the random seeds are not fixed.

    Returns
    -------
    train_X : 2-dimensional numpy array
        A numpy array holding the train data.
    test_X : 2-dimensional numpy array
        A numpy array holding the test data.
    train_y : 1-dimensional numpy array
        A numpy array holding labels of the train data.
    test_y : 1-dimensional numpy array
        A numpy array holding labels of the test data.
    """
    assert random_seed is None or isinstance(random_seed, int), 'Incorrect seed.'
    if random_seed is not None:
        import fatf
        fatf.setup_random_seed(random_seed)

    # Load Moons Dataset
    moons_data, moons_target = sklearn.datasets.make_moons(
        n_samples=1500, noise=0.25)

    # Scale it between 0 and 1
    scaler = sklearn.preprocessing.MinMaxScaler(feature_range=(0, 1))
    moons_data = scaler.fit_transform(moons_data)

    # Split into test and train data
    train_X, test_X, train_y, test_y = sklearn.model_selection.train_test_split(
          moons_data, moons_target, test_size=0.2)

    return train_X, test_X, train_y, test_y


def _download_bikes():
    """
    Downloads the UCI Bike Sharing data set and extracts a subset of its features.

    This function downloads the UCI Bike Sharing data set
    <https://archive.ics.uci.edu/ml/datasets/bike+sharing+dataset>
    and removes the following two features:

    * record index, and
    * date.

    Out of the three target variables (casual, registered and cnt),
    it selects *cnt*, which represents the total number of bikes rented
    during a given day.

    Returns
    -------
    bikes_data : 2-dimensional numpy array
        A numpy array holding the data.
    bikes_target : 1-dimensional numpy array
        A numpy array holding the target variable of the data (the number of
        bikes rented on a given day).
    bikes_feature_names : list of strings
        A Python list holding the feature names.
    bikes_target_name : string
        A string with the name of the target variable.
    """
    # Load Bikes
    url=('https://archive.ics.uci.edu/ml/machine-learning-databases/00275/'
         'Bike-Sharing-Dataset.zip')
    request = requests.get(url)
    request_io_stream = io.BytesIO(request.content)

    with zipfile.ZipFile(request_io_stream) as file:
        data = file.read('day.csv').decode('utf-8')

    # Do not read the first two columns (0: index, 1: date); and skip the two
    # penultimate ones (13: casual users count, 14: registered users count)
    # since the last one (15) is the total count.
    column_ids = (2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 15)
    _bikes_data = np.genfromtxt(
        io.StringIO(data),
        delimiter=',',
        skip_header=1,
        usecols=column_ids,
        dtype=np.float32)

    # Separate data from target
    ## Drop the target column
    bikes_data = np.delete(_bikes_data, -1, axis=1)
    ## Target is the total bike rental count
    bikes_target = _bikes_data[:, -1]

    # Get the feature names from the csv header
    _bikes_names = []
    for i, name in enumerate(data.split('\n')[0].split(',')):
        if i in column_ids:
            _bikes_names.append(name.strip())

    # Separate data from target
    bikes_feature_names = _bikes_names[:-1]
    bikes_target_name = _bikes_names[-1]

    return bikes_data, bikes_target, bikes_feature_names, bikes_target_name


def _preprocess_bikes_target(bikes_target):
    """
    Discretises the target variable of the UCI Bike Sharing data set.

    This function bins the regression target variable of the UCI Bike Sharing
    data set (see :func:`_download_bikes`) into three classes, making it a
    classification task:

    * ``0`` is *low*: 0 <= y < 4000;
    * ``1`` is *medium*: 4000 <= y < 6000; and
    * ``2`` is *high*: 6000 <= y < 9000.

    Parameters
    ----------
    bikes_target : 1-dimensional numpy array
        A numpy array holding the target variable of the UCI Bike Sharing data
        set (the number of bikes rented on a given day).

    Returns
    -------
    bikes_binned_target : 1-dimensional numpy array
        A discretised version of the ``bikes_target`` array.
    """
    bins = [0.0, 4000.0, 6000.0, 9000.0]
    # Subtract 1 to start the count from 0
    bikes_binned_target = np.digitize(bikes_target, bins=bins) - 1

    return bikes_binned_target


def generate_bikes(random_seed=None):
    """
    Generates the UCI Bike Sharing data set.

    This function downloads the Bike Sharing data set from the UCI repository
    and removes *record index* and *date* features
    (see :func:`_download_bikes`).
    It also discretises the target variable into three classes: *low* (0),
    *medium* (1) and *high* (2) -- see :func:`_preprocess_bikes_target`.

    For reproducibility of the train/test split, you may wish to set the
    ``random_seed`` parameter.

    Parameters
    ----------
    random_seed : integer, optional (default=None)
        A random seed used to initialise Python's and numpy's ``random``
        modules. If ``None``, the random seeds are not fixed.

    Returns
    -------
    train_X : 2-dimensional numpy array
        A numpy array holding the train data.
    test_X : 2-dimensional numpy array
        A numpy array holding the test data.
    train_y : 1-dimensional numpy array
        A numpy array holding labels of the train data.
    test_y : 1-dimensional numpy array
        A numpy array holding labels of the test data.
    bikes_feature_names : list of strings
        A Python list holding the feature names.
    bikes_target_name : string
        A string with the name of the target variable.
    """
    assert random_seed is None or isinstance(random_seed, int), 'Incorrect seed.'
    if random_seed is not None:
        import fatf
        fatf.setup_random_seed(random_seed)

    # Load Bikes
    bikes_data, bikes_target, bikes_feature_names, bikes_target_name = (
        _download_bikes()
    )

    # Convert the regression target into classification
    bikes_classification_target = _preprocess_bikes_target(bikes_target)

    # Split into test and train data
    train_X, test_X, train_y, test_y = sklearn.model_selection.train_test_split(
        bikes_data, bikes_classification_target,
        test_size=0.2,
        stratify=bikes_classification_target)

    return (train_X, test_X, train_y, test_y,
            bikes_feature_names, bikes_target_name)


def get_boston():
    """
    Generates the Boston Housing data set.

    The target variable is discretised into two classes: *low* (0) and
    *high* (1).
    The discretisation threshold is fixed at 20.

    Returns
    -------
    X : 2-dimensional numpy array
        A numpy array holding the data.
    y_class : 1-dimensional numpy array
        A numpy array holding discretised labels of the data.
    """
    X, y = sklearn.datasets.load_boston(return_X_y=True)
    y_class = np.zeros_like(y, dtype=np.int8)
    y_class[y >= 20] = 1

    return X, y_class

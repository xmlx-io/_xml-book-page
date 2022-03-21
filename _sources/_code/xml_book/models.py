"""
XML Book Models Module
======================

This module implements a collection of predictive model functions used by
the book.
"""

# Author: Kacper Sokol <kacper@xmlx.io>
#         Alex Hepburn <ah13558@bristol.ac.uk>
# License: MIT

import sklearn.ensemble
import sklearn.svm

__all__ = ['get_random_forest', 'get_svc']


def get_random_forest(data, target, random_seed=None):
    """
    Fits a Random Forest classifier.

    This function fits a Random Forest classifier using the scikit-learn's
    ``sklearn.ensemble.RandomForestClassifier`` class.

    For reproducibility of the model, you may wish to set the
    ``random_seed`` parameter.

    Parameters
    ----------
    data : 2-dimensional numpy array
        Training data of the model.
    target : 1-dimensional numpy array
        Target variable (classification) for the training data of the model.
    random_seed : integer, optional (default=None)
        A random seed used to initialise Python's and numpy's ``random``
        modules as well as scikit-learn's ``random_state`` parameter.
        If ``None``, the random seeds are not fixed.

    Returns
    -------
    clf : sklearn.ensemble.RandomForestClassifier
        A fitted Random Forest classifier.
    """
    assert random_seed is None or isinstance(random_seed, int), 'Incorrect seed.'
    if random_seed is not None:
        import fatf
        fatf.setup_random_seed(random_seed)

    clf = sklearn.ensemble.RandomForestClassifier(
        n_estimators=5, max_depth=7, random_state=random_seed)
    clf.fit(data, target)

    return clf


def get_svc(data, target, random_seed=None):
    """
    Fits a Support Vector Machine classifier.

    This function fits a Support Vector Machine classifier using the
    scikit-learn's ``sklearn.svm.SVC`` class.

    For reproducibility of the model, you may wish to set the
    ``random_seed`` parameter.

    Parameters
    ----------
    data : 2-dimensional numpy array
        Training data of the model.
    target : 1-dimensional numpy array
        Target variable for the training data of the model.
    random_seed : integer, optional (default=None)
        A random seed used to initialise Python's and numpy's ``random``
        modules as well as scikit-learn's ``random_state`` parameter.
        If ``None``, the random seeds are not fixed.

    Returns
    -------
    clf : sklearn.ensemble.RandomForestClassifier
        A fitted Support Vector Machine classifier.
    """
    assert random_seed is None or isinstance(random_seed, int), 'Incorrect seed.'
    if random_seed is not None:
        import fatf
        fatf.setup_random_seed(random_seed)

    clf = sklearn.svm.SVC(probability=False, random_state=random_seed)
    clf.fit(data, target)

    return clf

from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
import numpy as np


# Position match metrics

def norm_kendall(r1, r2):
    """
    Compute the normalized Kendall tau distance between two lists.

    Parameters:
    - r1: list, first list to compare.
    - r2: list, second list to compare.

    Returns:
    - Normalized Kendall tau distance between r1 and r2.
    """
    n = len(r1)
    assert len(r2) == n, "Both lists have to be of equal length"
    i, j = np.meshgrid(np.arange(n), np.arange(n))
    a = np.argsort(r1)
    b = np.argsort(r2)
    ndisordered = np.logical_or(np.logical_and(a[i] < a[j], b[i] > b[j]), np.logical_and(a[i] > a[j], b[i] < b[j])).sum()
    return (ndisordered / (n * (n - 1))).round(4)

def position_match_rate(r1, r2):
    """
    Position match rate

    Parameters:
    - r1: list, first list to compare.
    - r2: list, second list to compare.

    Returns:
    float: The normalized number of elements in the same position in the two lists.
    """
    assert len(r1) == len(r2), "Both lists have to be of equal length"
    same_position = 0
    for i in range(len(r1)):
        if r1[i] == r2[i]:
            same_position += 1
    return same_position/len(r1)

def dbs(rank_mcdm, rank_ml):
    # From: https://doi.org/10.1016/j.eswa.2020.113527
    """
    Calculates the distance (in positions) between the ML solution and MCDM top-1 solution.

    Parameters:
    rank_ml (list or array): The ML solution.
    rank_mcdm (list or array): The best MCDM solution.

    Returns:
    int: The difference between the position of the 1st element of list 1 and the same element in list 2.
    """
    if len(rank_mcdm) != len(rank_ml):
        raise ValueError("Both lists should have the same size.")
    return rank_ml.index(rank_mcdm[0]) - rank_mcdm.index(rank_mcdm[0])


# Regression Loss Metrics
# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_absolute_percentage_error.html
def mape(y_pred, y_true):
    return mean_absolute_percentage_error(y_true, y_pred,
                                          multioutput='uniform_average')
# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html
def mse(y_pred, y_true):
    return mean_squared_error(y_true, y_pred,
                              multioutput='uniform_average',
                              squared=True)
# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html
def rmse(y_pred, y_true):
    return mean_squared_error(y_true, y_pred,
                              multioutput='uniform_average',
                              squared=False)
# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html
def r2(y_pred, y_true):
    return r2_score(y_true, y_pred,
                    multioutput='uniform_average')

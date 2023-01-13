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

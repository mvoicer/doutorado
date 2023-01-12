import numpy as np


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

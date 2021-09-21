import scipy.stats as stats
import numpy as np


class Similarity:
    def __init__(self, rank1, rank2):
        self.rank1 = rank1
        self.rank2 = rank2

    def kendall(self):
        """
        Values close to 1 indicate strong agreement,
        and values close to -1 indicate strong disagreement.
        :param r1: list 1
        :param r2: list 2
        :return: level of agreement between two lists
        """
        tau, p_value = stats.kendalltau(self.rank1, self.rank2)
        return tau

    def norm_kendall(self):
        """
        Compute the Kendall tau distance.
        """
        n = len(self.rank1)
        assert len(self.rank2) == n, "Both lists have to be of equal length"
        i, j = np.meshgrid(np.arange(n), np.arange(n))
        a = np.argsort(self.rank1)
        b = np.argsort(self.rank2)
        ndisordered = np.logical_or(np.logical_and(a[i] < a[j], b[i] > b[j]),
                                    np.logical_and(a[i] > a[j], b[i] < b[j])).sum()
        return (ndisordered / (n * (n - 1))).round(4)


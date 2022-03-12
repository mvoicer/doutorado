import pandas as pd
import numpy as np


class Normalization:
    # def __init__(self, matrix, cb, weights):
    #     if matrix.shape[1] != len(weights) or matrix.shape[1] != len(cb) or len(cb) != len(weights):
    #         raise ValueError(f'Data shape, cost-benefit vector or weights vector does not match')
    #     self.matrix = matrix
    #     self.cb = cb
    #     self.weights = weights

    def __init__(self, matrix, cb):
        if matrix.shape[1] != len(cb):
            raise ValueError(f'Data shape and cost-benefit vector does not match')
        self.matrix = matrix
        self.cb = cb

    def normalization_zero_one(self):
        df = self.matrix.to_numpy()
        dff = np.zeros(self.matrix.shape)
        for i in range(len(self.cb)):
            X = df[:, i]
            if self.cb[i] == 'cost':
                # lower is better
                dff[:, i] = (X.max() - X) / (X.max() - X.min())
            else:
                # greater is better
                dff[:, i] = (X - X.min()) / (X.max() - X.min())
        return pd.DataFrame(dff).round(4)

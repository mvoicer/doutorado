import pandas as pd
import numpy as np


class Normalization:

    def __init__(self, matrix, nobj, cb):
        if nobj != len(cb):
            raise ValueError(f'Data shape and cost-benefit vector does not match')
        if cb is None:
            self.cb = ['cost'] * nobj
        else:
            self.cb = cb
        self.matrix = matrix

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

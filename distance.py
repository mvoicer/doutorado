from scipy.spatial import distance
import pandas as pd
import numpy as np


class Distance:
    def __init__(self, df):
        self.df = df

    def euclidean(self):
        """
        Calculate the euclidean distance among all the solutions.
        Input: Pareto front solutions
        Output: Squared matrix with the distance among the solutions.
        """
        # TODO: optimize code - não é necessário calcular a diagnonal e nem os dois lados da matriz
        df_aux = pd.DataFrame(np.zeros((self.df.shape[0], self.df.shape[0])))
        for r in range(self.df.shape[0]):
            for c in range(self.df.shape[0]):
                df_aux.loc[r, c] = distance.euclidean(self.df.iloc[r], self.df.iloc[c])
        return df_aux

    def cosine(self):
        """
        Compute the cosine similarity
        :return: cosine similarity in [-1 1] interval.
        """
        df_aux = pd.DataFrame(np.zeros((self.df.shape[0], self.df.shape[0])))
        for r in range(self.df.shape[0]):
            for c in range(self.df.shape[0]):
                df_aux.loc[r, c] = 1 - distance.cosine(self.df.iloc[r], self.df.iloc[c])
        return df_aux



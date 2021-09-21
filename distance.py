from scipy.spatial import distance
import pandas as pd
import numpy as np


class Distance:
    def __init__(self, df):
        self.df = df

    def rec_euclidean(self):
        """
        Calculate the euclidean distance among all the solutions. CODE NEED TO BE OPTIMIZED.
        Input: Pareto front solutions
        Output: Squared matrix with the distance among the solutions.
        """
        df_aux = pd.DataFrame(np.zeros((self.df.shape[0], self.df.shape[0])))
        for r in range(self.df.shape[0]):
            for c in range(self.df.shape[0]):
                df_aux.loc[r, c] = distance.euclidean(self.df.iloc[r], self.df.iloc[c])
        return df_aux

    # def cosine(self):
    #     """
    #     Compute the cosine similarity
    #     :return: cosine similarity
    #     """
    #     return 1 - spatial.distance.cosine(self.rank1, self.rank2)

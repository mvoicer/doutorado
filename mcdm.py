from normalization import Normalization
import pandas as pd
import numpy as np


class Mcdm:
    def __init__(self, df, weights, cb):
        self.df = df
        self.weights = weights
        self.cb = cb

    def promethee_ii(self):
        nrow = self.df.shape[0]
        ncol = self.df.shape[1]

        # Normalization
        ndm = Normalization(self.df, self.cb, self.weights).normalization_zero_one()

        # Pairwise comparisons
        pc_matrix = pd.DataFrame(np.zeros((nrow ** 2, ncol)))
        aux = 0
        for r in range(ndm.shape[0]):
            for s in range(ndm.shape[0]):
                pc_matrix.iloc[aux][:] = ndm.iloc[r, :] - ndm.iloc[s, :]
                aux += 1

        # Reshape the PC matrix
        pc_matrix_shaped = pd.DataFrame()
        for i in range(pc_matrix.shape[1]):
            sol = pd.DataFrame(np.hstack(np.split(pc_matrix.loc[:, i], nrow)).reshape(nrow, nrow)).T
            pc_matrix_shaped = pd.concat([pc_matrix_shaped, sol], axis=1)


        # Calculate the preference function
        pref_func_matrix = pc_matrix.copy()
        pref_func_matrix[pref_func_matrix < 0] = 0

        # Calculate the aggregated preference function
        agg_pref_func_matrix = (pref_func_matrix * self.weights).apply(sum, axis=1)
        agg_pref_func_matrix = pd.DataFrame(agg_pref_func_matrix.values.reshape(nrow, nrow))

        # Determine the leaving and entering outranking flows
        outranking_flows = pd.DataFrame(np.zeros((nrow, 2)), columns=['Leaving', 'Entering'])
        outranking_flows['Leaving'] = agg_pref_func_matrix.apply(sum, axis=1) / (ncol - 1)
        outranking_flows['Entering'] = agg_pref_func_matrix.apply(sum, axis=0) / (ncol - 1)
        outranking_flows['Leav_Enter'] = outranking_flows['Leaving'] - outranking_flows['Entering']

        # Determine the ranking
        outranking_flows['Rank'] = outranking_flows['Leav_Enter'].rank(ascending=False).astype(int)

        # Return the pairwise comparison matrix and the ranking as a list
        return pc_matrix_shaped, outranking_flows['Rank'].to_list()

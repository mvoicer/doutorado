from normalization import Normalization
import pandas as pd
import numpy as np


class Gera_Pc_Mcdm:
    def __init__(self, df, weights=None, cb=None):
        self.df = df
        if weights is None:
            self.weights = [1 / df.shape[1]] * df.shape[1]
        else:
            self.weights = weights
        if cb is None:
            self.cb = ['cost'] * df.shape[1]
        else:
            self.cb = cb

    def promethee_ii_pc(self):
        # Followed steps of Prof. Manoj Mathew on https://www.youtube.com/watch?v=xe2XgGrI0Sg

        nrow, ncol = self.df.shape
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

        return pc_matrix_shaped, pc_matrix

    def ahp_pc(self):
        # Calculate the AHP based on the values of each alternative/objective
        min_col = self.df.min(axis=0).to_list()
        max_col = self.df.max(axis=0).to_list()
        nrow, ncol = self.df.shape

        pc_matrix = pd.DataFrame()

        for col in range(ncol):
            # Calculate intervals equally divided into 1-9
            intervals = np.linspace(0, max_col[col] - min_col[col], num=9)

            # Calculate the difference among all solutions
            m1 = np.tile(self.df.iloc[:, col], (nrow, 1))
            m2 = np.tile(self.df.iloc[:, col].to_numpy().reshape(-1, 1), (1, nrow))
            df_dif = pd.DataFrame(m2 - m1)

            new_df_dif = df_dif.copy()

            # Replace the diff values by the Saaty's scale values
            for i in range(nrow):
                for j in range(nrow):
                    if i == j:
                        new_df_dif.iloc[i, j] = 1
                    elif df_dif.iloc[i, j] < 0:
                        continue
                    else:
                        for idx, valor in enumerate(intervals):
                            if df_dif.iloc[i, j] <= valor:
                                new_df_dif.iloc[j, i] = idx + 1
                                new_df_dif.iloc[i, j] = 1 / (idx + 1)
                                break

            # If cost criteria: inverse of the assigned number
            if self.cb[col] == 'cost':
                new_df_dif = 1 / new_df_dif
            else:
                pass

            # Concat the pairwise matrices
            pc_matrix = pd.concat([pc_matrix, new_df_dif], axis=1)

        # return pc_matrix
        return pc_matrix

class Mcdm_ranking:
    def __init__(self):
        pass

    def promethee_ii_ranking(self, pc_matrix, weights=None, nobj=None, nrow=None):
        if weights is None:
            weights = [1 / nobj] * nobj
        else:
            weights = weights
        # Calculate the preference function
        pc_matrix[pc_matrix <= 0] = 0

        # Calculate the aggregated preference function
        _agg_pref_func_matrix = pc_matrix * weights
        _agg_pref_func_matrix = _agg_pref_func_matrix.apply('sum', axis=1)
        agg_pref_func_matrix = pd.DataFrame(_agg_pref_func_matrix.values.reshape(nrow, nrow))

        # Determine the leaving and entering outranking flows
        outranking_flows = pd.DataFrame(np.zeros((nrow, 2)), columns=['Leaving', 'Entering'])
        outranking_flows['Leaving'] = agg_pref_func_matrix.apply(sum, axis=1) / (nobj - 1)
        outranking_flows['Entering'] = agg_pref_func_matrix.apply(sum, axis=0) / (nrow - 1)
        outranking_flows['Leav_Enter'] = outranking_flows['Leaving'] - outranking_flows['Entering']

        # Determine the ranking
        ranking = outranking_flows['Leav_Enter'].sort_values(ascending=False).index.to_list()

        return ranking

    def ahp_ranking(self, pc_matrix, weights=None, nrow=None, nobj=None):
        if weights is None:
            weights = [1 / nobj] * nobj
        else:
            weights = weights

        # Priority vector
        eigen = pd.DataFrame()

        temp = nrow
        # Percorre a matriz de comparacoes pareadas e calcula as prioridades de cada matriz (i.e. da PC de cada objetivo)
        # e no final (eigen) concatena elas para, entao, multiplicar pelos pesos e calcular o ranking.
        for i in range(0, pc_matrix.shape[1], nrow):
            _pc_matrix = pc_matrix.iloc[:, i:temp]
            temp += nrow

            eigen_daqui = (_pc_matrix / (_pc_matrix.apply('sum', axis=0))).apply('sum', axis=1) / nrow
            eigen = pd.concat([eigen, eigen_daqui], axis=1)

        # Calculate ranking
        ranking = (eigen * weights).apply('sum', axis=1).sort_values(ascending=True).index.to_list()

        return ranking

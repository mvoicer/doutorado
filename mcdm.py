from normalization import Normalization
import pandas as pd
import numpy as np

class Gera_Pc_Mcdm:
    def __init__(self, df, cb):
        self.df = df
        self.cb = cb

    def promethee_ii_pc(self):
        # Followed steps of Prof. Manoj Mathew on https://www.youtube.com/watch?v=xe2XgGrI0Sg
        nrow, ncol = self.df.shape

        # Normalization
        ndm = Normalization(self.df, self.cb).normalization_zero_one()

        # Pairwise comparisons
        pc_matrix = pd.DataFrame(np.zeros((nrow ** 2, ncol)))
        aux = 0
        for r in range(ndm.shape[0]):
            for s in range(ndm.shape[0]):
                pc_matrix.iloc[aux][:] = ndm.iloc[r, :] - ndm.iloc[s, :]
                aux += 1

        # # Reshape the PC matrix to squared matrix
        pc_matrix_shaped = pd.DataFrame()
        for i in range(pc_matrix.shape[1]):
            sol = pd.DataFrame(np.hstack(np.split(pc_matrix.loc[:, i], nrow)).reshape(nrow, nrow)).T
            pc_matrix_shaped = pd.concat([pc_matrix_shaped, sol], axis=1)

        # # Calculate the preference function Pj(a,b)
        # pc_matrix_shaped[pc_matrix_shaped <= 0] = 0

        return pc_matrix_shaped

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
            df_dif = pd.DataFrame(np.tile(self.df.iloc[:, col].to_numpy().reshape(-1, 1), (1, nrow)) - \
                                  np.tile(self.df.iloc[:, col], (nrow, 1)))

            # Replace the diff values by the Saaty's scale values
            new_df_dif = df_dif.copy()
            for i in range(nrow):
                for j in range(nrow):
                    if df_dif.iloc[i, j] < 0:
                        continue
                    else:
                        for idx, valor in enumerate(intervals):
                            if df_dif.iloc[i, j] <= valor:
                                new_df_dif.iloc[j, i] = idx + 1
                                # new_df_dif.iloc[i, j] = 1 / (idx + 1)     #utiliza fração na escala Saaty
                                new_df_dif.iloc[i, j] = -(idx + 1)          # utiliza valor negativo na escala Saaty
                                break

            # If cost criteria: inverse of the assigned number
            if self.cb[col] == 'cost':
                # new_df_dif = 1 / new_df_dif   #utiliza fração na escala Saaty
                new_df_dif = -new_df_dif        # utiliza valor negativo na escala Saaty
            else:
                pass

            np.fill_diagonal(new_df_dif.to_numpy(), 1)  # preenche diagonal principal com 1

            # Concat the pairwise matrices
            pc_matrix = pd.concat([pc_matrix, new_df_dif], axis=1)

        return pc_matrix

class Mcdm_ranking:
    def __init__(self):
        pass

    def promethee_ii_ranking(self, pc_matrix, weights, nobj, nrow):
        weights = [1 / nobj] * nobj if weights is None else weights

        # Calculate the preference function Pj(a,b)
        pc_matrix[pc_matrix <= 0] = 0

        ## Reshape the matrix to multiply the weights: n x n -> n**2 x nobj
        temp_ = pd.DataFrame()
        t = nrow
        for j in range(0, pc_matrix.shape[1], nrow):
            temp = pd.DataFrame()
            t += j
            prov = pc_matrix.iloc[:, j:t]

            for i in range(nrow):
                temp = pd.concat([temp, prov.iloc[:, i]], axis=0)
            temp_ = pd.concat([temp_, temp], axis=1)
            del temp

        ## Apply the multiplication
        _agg_pref_func_matrix = temp_ * weights

        # Aggregated preference function pi(a,b)
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

    def ahp_ranking(self, pc_matrix, weights, nobj, nrow, cb):
        eigen = pd.DataFrame()
        temp = nrow

        # Retorna os valores para a escala original
        pc_matrix[pc_matrix < 0] = 1 / np.abs(pc_matrix)

        weights = [1 / len(cb)] * len(cb) if weights is None else weights


        # Percorre a matriz de comparacoes pareadas e calcula as prioridades de cada matriz (i.e. da PC de cada objetivo)
        # e no final (eigen) concatena elas para, entao, multiplicar pelos pesos e calcular o ranking.
        # Ver: https://ricardo-vargas.com/pt/articles/analytic-hierarchy-process/
        # Ver: https://edisciplinas.usp.br/mod/resource/view.php?id=2822802
        for i in range(0, pc_matrix.shape[1], nrow):
            _pc_matrix = pc_matrix.iloc[:, i:temp]
            temp += nrow

            # normalized matrix and Eigen vector
            _pc_matrix = _pc_matrix / _pc_matrix.sum()

            # weights vector (w_ij)
            # Normalize the values of the matrix by dividing each element by the sum of the column to which it belongs.
            # The "relative priority" (RP) column in the following table shows the average of each criterion.
            relative_priority = _pc_matrix.sum(axis=1) / nrow

            # # to find relative weight of each row by dividing the sum of the values of each row of n.
            # consistency_vector = _pc_matrix.sum(axis=1) / relative_priority
            #
            # # lambda_max is equal to the sum of the elements of the column vector AW.
            # # highest eigenvalue lambda_max of the parity matrix is obtained by means of the arithmetic
            # # mean of the elements of the consistency vector.
            # lambda_max = np.sum(consistency_vector) / nrow
            #
            # # consistency index (CI) - calculado via slide da USP
            # ci_i = (lambda_max - nrow) / (nrow - 1)

            # Concatena vetor de prioridades
            eigen = pd.concat([eigen, relative_priority], axis=1, ignore_index=True)

        # Calculate ranking
        ranking = (eigen * weights).apply('sum', axis=1).sort_values(ascending=True).index.to_list()
        return ranking

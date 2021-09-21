import pandas as pd
import numpy as np


def create_subsample(df_var, df_pref, nobj, index):
    """
    Create sub-dataframes with the features (alternatives) and target (value in the objective space).
    :param df_var:
    :param df_pref:
    :param nobj:
    :param index:
    :return:
    """

    # Create a df_aux that receive the features concatenated (objectives) and targets (preference)
    sub_df = pd.DataFrame(np.zeros((len(index), df_var.shape[1] * 2 + nobj)))
    cont = 0
    for i in index:
        for j in index:
            # Concatenate the two rows - i.e. values of the objectives
            # and the preference between the two objectives
            sub_df.loc[cont] = pd.concat([df_var.loc[i], df_var.loc[j], df_pref.loc[i, j]], axis=0, ignore_index=True)
            cont += 1
    return sub_df


def merge_matrices(idx_N_Q, preference_matrix, ml_predicted, nobj, npop):
    """
    Replace the predicted values in the preference matrix to calculate
    if the rankings (predicted vs preference) are equal or not.
    :param idx_N_Q: N-Q index
    :param preference_matrix: preference matrix
    :param ml_predicted: ranking obtained with the ML method
    :param nobj: number of objectives
    :param npop: number of solutions
    :return: dataframe merged with the real values and the values predicted
    """
    df_merged = preference_matrix.copy()
    for col in range(nobj):
        row = 0
    for s1 in idx_N_Q:
        for s2 in idx_N_Q:
            df_merged.iloc[s1, s2 + npop * col] = ml_predicted.loc[row, col]
            row += 1
    return df_merged

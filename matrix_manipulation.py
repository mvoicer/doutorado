import pandas as pd
import numpy as np


def create_subsample(df_var, df_pref, nobj, index, all_comb):
    """
    Create sub-dataframes with the features (alternatives) and target (value in the objective space).
    """
    sub_df = pd.DataFrame(np.zeros((len(index), df_var.shape[1] * 2 + nobj)))
    cont = 0
    if all_comb is True:
        for i in index:
            for j in index:
                # Concatenate the two rows - i.e. values of the objectives
                # and the preference between the two objectives
                sub_df.loc[cont] = pd.concat([df_var.loc[i], df_var.loc[j], df_pref.loc[i, j]], axis=0,
                                             ignore_index=True)
                cont += 1
    else:
        for i in index:
            for j in index:
                if i != j:
                    sub_df.loc[cont] = pd.concat([df_var.loc[i], df_var.loc[j], df_pref.loc[i, j]], axis=0,
                                                 ignore_index=True)
                else:
                    continue
                cont += 1
    X = sub_df.iloc[:, :-nobj]
    y = sub_df.iloc[:, -nobj:]
    return X, y


def merge_matrices(idx_N_Q, preference_matrix, ml_predicted, all_comb):
    """
    Replace the predicted values in the preference matrix to calculate
    if the rankings (predicted vs preference) are equal or not.
    :param idx_N_Q: N-Q index
    :param preference_matrix: preference matrix
    :param ml_predicted: ranking obtained with the ML method
    :return: dataframe merged with the real values and the predicted values
    """
    df_merged = preference_matrix.copy()

    comb_idx = []
    if all_comb is True:
        # Gera todas as combinações do N-Q
        for i in idx_N_Q:
            for k in idx_N_Q:
                comb_idx.append(tuple([i, k]))
    else:
        # Generate the combinations with the N_Q indexes
        comb_idx = []
        for i in idx_N_Q:
            for k in idx_N_Q:
                if i != k:
                    comb_idx.append(tuple([i, k]))

    results = pd.DataFrame()
    x = 0
    for k in range(0, df_merged.shape[1], df_merged.shape[0]):
        m = df_merged.iloc[:, k:k+df_merged.shape[0]].to_numpy()

        for i, idx in enumerate(comb_idx):
            m[idx] = ml_predicted.values[i, x]
        x += 1
        m = pd.DataFrame(m)
        results = pd.concat([results, m], ignore_index=False, axis=1)
    return results

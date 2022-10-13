import pandas as pd
import numpy as np


def create_subsample(df_var, df_pref, nobj, index):
    """
    Summary:
        Create sub-dataframes with the features (alternatives) and target (value in the objective space).
    Parameters:
        df_var: dataframe with the variables in decision space
        df_pref: preference matrix
        nobj: number of objectives
        index: index of the selected alternatives
    Returns:
        dataframe used as input for ML methods. Concatenation of the decision variables and preferences (target)
    """
    sub_df = pd.DataFrame(np.zeros((len(index), df_var.shape[1] * 2 + nobj)))
    cont = 0
    for i in index:
        for j in index:
            if i != j:
                sub_df.loc[cont] = pd.concat([df_var.loc[i], df_var.loc[j], df_pref.loc[i, j]], axis=0, ignore_index=True)
            else:
                continue
            cont += 1
    X = sub_df.iloc[:, :-nobj]
    y = sub_df.iloc[:, -nobj:]
    return X, y


def merge_matrices(idx_N_Q, preference_matrix, ml_predicted):
    """
    Summary:
        Replace the predicted values in the preference matrix to calculate if the rankings (predicted vs preference)
        are equal or not.
    Parameters:
        idx_N_Q: N-Q index
        preference_matrix: preference matrix
        ml_predicted: y_pred (predicted) from the ML method
    Returns:
        dataframe merged with the real values and the predicted values
    """
    df_merged = preference_matrix.copy()

    # Gera as combinações do N-Q, exceto diagnonal principal
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

import pandas as pd
import numpy as np


def condorcet(Q, entrance, pc_matrix, df_var, nobj):
    """
    It is a kind of regularization. It is based on the known preferences among the alternatives listed.
    For each objective, measure the difference between Ai and Aj (Q) and Aj and Aq (recommended) and infer the
    preference between Ai and Aq.
    E.g. Ai > Aj = 9 and Aj > Aq = 3. Then, Ai > Aq = 6 (9-3).
    If the inferred preference is greater than 0, keep it. Otherwise, 1/pref.
    """
    rows = (len(Q) - 1) * len(entrance)
    cols = (df_var.shape[1] * 2) + nobj
    df_cond = np.zeros((rows, cols))

    z = 0
    for x in Q[:-1]:
        for y in entrance:
            diff = []
            temp = []
            # Differences in Q
            arr1 = pc_matrix.loc[x, Q[-1]]
            # Differences in the entrance alternatives
            arr2 = pc_matrix.loc[Q[-1], y]
            # Calculate the differences
            [diff.append(arr1 - arr2) for arr1, arr2 in zip(arr1, arr2)]
            # print("Diff entre", x, Q[-1], 'e', Q[-1], y, 'é: ', diff)
            [temp.append(i) if i > 0 else temp.append(1 / np.abs(i)) for i in diff]

            # Merge the solutions in the variables space
            joined = np.hstack([df_var.loc[x].T, df_var.loc[y].T]).tolist()

            # Merge the variables and target
            df_cond[z] = joined+temp
            z += 1
    df_t = pd.DataFrame(df_cond)

    # Remove infinite values from the dataset, if it exists
    df = df_t[np.isfinite(df_t).all(1)]

    X = df.iloc[:, :-nobj]
    y = df.iloc[:, -nobj:]

    return X, y


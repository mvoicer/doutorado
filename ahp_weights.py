import numpy as np
import pandas as pd


def calculate_ahp_weights(df_obj):
    df = df_obj.sum()
    min_col = df.min(axis=0)
    max_col = df.max(axis=0)
    nrow = df.shape[0]
    intervals = np.linspace(0, max_col - min_col, num=9)
    df_dif = pd.DataFrame(np.tile(df.to_numpy().reshape(-1, 1), (1, nrow)) - np.tile(df, (nrow, 1)))

    # Automatiza o cálculo dos pesos calculando os scores numa escala de [1-9].
    new_df_dif = df_dif.copy()
    for i in range(nrow):
        for j in range(nrow):
            if df_dif.iloc[i, j] < 0:
                continue
            else:
                for idx, valor in enumerate(intervals):
                    if df_dif.iloc[i, j] <= valor:
                        new_df_dif.iloc[j, i] = idx + 1
                        new_df_dif.iloc[i, j] = -(idx + 1)
                        break
    np.fill_diagonal(new_df_dif.to_numpy(), 1)
    new_df_dif[new_df_dif < 0] = 1 / np.abs(new_df_dif)
    weights = new_df_dif.mean(1).to_numpy()
    print("AHP weights calculado: ", weights)
    return weights

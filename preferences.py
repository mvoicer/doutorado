import numpy as np
import pandas as pd


# Define Saaty's 1-9 Scale for AHP Preference
def matrix_intervalos(A):
    max_A = np.max(A, axis=0)
    min_A = np.min(A, axis=0)
    I = np.zeros(shape=(10, len(max_A)))
    for i, (menor, maior) in enumerate(zip(min_A, max_A)):
        intervalos = np.linspace(menor, maior, 10)
        I[:, i] = intervalos.ravel()
    return I


# Calculate the differences among the nominal values of the objectives
def my_cdist(df_obj):
    """
     Calculate the differences among the nominal values of the objectives
    :param df_obj: dataframe with the values in the objective space
    :return:
    """
    npop = df_obj.shape[0]
    m1 = np.tile(df_obj, (npop, 1))
    m2 = np.tile(df_obj.reshape(-1, 1), (1, npop))
    return m2 - m1


def preferencia(df_dif, interval):
    """
    Calculate the preferences
    :param df_dif:
    :param interval:
    :return:
    """
    df_pref = np.ones(shape=df_dif.shape)

    it = np.nditer(df_dif, flags=['multi_index'])

    for x in it:
        for j, _ in enumerate(interval):
            if j == len(interval):
                df_pref[it.multi_index] = 9 if x > 0 else 1.0 / 9.0
                # df_pref[it.multi_index] = 1.0 / 9.0 if x > 0 else 9
                break

            if interval[j] <= np.abs(x) <= interval[j + 1]:
                df_pref[it.multi_index] = 1.0 / (j + 1) if x > 0 else j + 1
                break
    return df_pref.round(3)


def notas_pref(A):
    df_pref = pd.DataFrame()
    I = matrix_intervalos(A)

    # For each objective
    for i, sol in enumerate(A.T):
        # Calculate the difference among the values
        df_dif = my_cdist(sol)
        # Get the intervals
        interval = I[:, i]
        # Generate the PC matrices
        pp = pd.DataFrame(preferencia(df_dif, interval), index=None)

        df_pref = pd.concat([df_pref, pp], axis=1)

    return df_pref

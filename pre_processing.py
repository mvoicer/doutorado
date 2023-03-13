import pandas as pd
from mcdm import Gera_Pc_Mcdm, Mcdm_ranking
import numpy as np
from params import metrics

# Load dataframe e remove duplicados
def load_dataset(df):
    df_var = pd.read_csv('data/'+str(df)+'_DEC.CSV', header=None)  # decision variables
    df_var = df_var.round(5)
    df_obj = pd.read_csv('data/'+str(df)+'_OBJ.CSV', header=None)  # values in Pareto front
    df_obj = df_obj.round(5)
    n_var = df_var.shape[1]
    n_obj = df_obj.shape[1]
    df_merged = pd.concat([df_var, df_obj], ignore_index=True, sort=False, axis=1)
    df_merged.drop_duplicates(inplace=True)
    df_merged = df_merged.reset_index(drop=True)
    df_var = df_merged.iloc[:, :n_var]
    df_obj = df_merged.iloc[:, n_var:]
    df_obj.columns = np.arange(1, n_obj + 1)
    df_obj.columns = ["Obj " + str(i) for i in df_obj.columns]
    return df_var, df_obj


# Initialize a dict for the results

def initialize_results():
    return {metr: [] for metr in metrics}


# Generate the preferences
def generate_preferences(mcdm_method, df_obj, weights, cb):
    npop, nobj = df_obj.shape
    if weights is None:
        weights = [1 / nobj] * nobj
    else:
        weights = weights
    if cb is None:
        cb = ['cost'] * nobj
    else:
        cb = cb

    if mcdm_method == 'AHP':
        pc_matrix = Gera_Pc_Mcdm(df=df_obj, cb=cb).ahp_pc()
        rank_mcdm = Mcdm_ranking().ahp_ranking(pc_matrix=pc_matrix, weights=weights, nrow=npop)
    elif mcdm_method == 'Promethee':
        pc_matrix = Gera_Pc_Mcdm(df=df_obj, cb=cb).promethee_ii_pc()
        rank_mcdm = Mcdm_ranking().promethee_ii_ranking(pc_matrix=pc_matrix, weights=weights, nobj=nobj, nrow=npop)
    else:
        raise ValueError('Multicriteria method: {} not implemented'.format(mcdm_method))
    return pc_matrix, rank_mcdm


def calculate_mcdm_ranking(mcdm_method, df_merged, weights, npop, nobj, cb):
    if mcdm_method == 'AHP':
        rank_predicted = Mcdm_ranking().ahp_ranking(pc_matrix=df_merged, weights=weights, nrow=npop)
    elif mcdm_method == 'Promethee':
        rank_predicted = Mcdm_ranking().promethee_ii_ranking(pc_matrix=df_merged, weights=weights, nobj=nobj, nrow=npop)
    else:
        raise ValueError('Multicriteria method: {} not implemented'.format(mcdm_method))
    return rank_predicted

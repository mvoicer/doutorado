import pandas as pd
from distance import Distance
from mcdm import Gera_Pc_Mcdm, Mcdm_ranking
from params import *

# Load dataframe
def load_dataset(df, n_samples):
    df_var = pd.read_csv('data/'+str(df)+'_DEC.CSV', header=None)  # decision variables
    df_var = df_var.iloc[0:n_samples, :].round(5)
    df_obj = pd.read_csv('data/'+str(df)+'_OBJ.CSV', header=None)  # values in Pareto front
    df_obj = df_obj.iloc[0:n_samples, :].round(5)
    return df_var, df_obj


def calculate_similarities(df_obj, simm_approach):
    """
    Summary:
        Calculate the distance based on the 'simm_approach' among solutions
    Parameters:
        df_obj: dataframe with the values in objective space
        simm_approach:
            cos: cosine measure
            euc: euclidean distance
    Returns:
        dataframe with the distances
    """
    if simm_approach == 'cos':
        df_dist = Distance(df_obj).cosine()
    elif simm_approach == 'euc':
        df_dist = Distance(df_obj).euclidean()
    else:
        raise ValueError('Recommendation {} not implemented'.format(simm_approach))
    return df_dist

# Initialize a dict for the results
def initialize_results():
    results = {}
    for metr in list_metrics:
        results[metr] = []
    return results

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
        rank_mcdm = Mcdm_ranking().ahp_ranking(pc_matrix=pc_matrix, weights=weights, nrow=npop, nobj=nobj)
    elif mcdm_method == 'Promethee':
        pc_matrix = Gera_Pc_Mcdm(df=df_obj, cb=cb).promethee_ii_pc()
        rank_mcdm = Mcdm_ranking().promethee_ii_ranking(pc_matrix=pc_matrix, weights=weights, nobj=nobj, nrow=npop)
    else:
        raise ValueError('Multicriteria method: {} not implemented'.format(mcdm_method))
    return pc_matrix, rank_mcdm


def calculate_new_ranking(mcdm_method, df_merged, weights, npop, nobj):
    if mcdm_method == 'AHP':
        rank_predicted = Mcdm_ranking().ahp_ranking(pc_matrix=df_merged, weights=weights, nrow=npop, nobj=nobj)
    elif mcdm_method == 'Promethee':
        rank_predicted = Mcdm_ranking().promethee_ii_ranking(pc_matrix=df_merged, weights=weights, nobj=nobj, nrow=npop)
    else:
        raise ValueError('Multicriteria method: {} not implemented'.format(mcdm_method))
    return rank_predicted

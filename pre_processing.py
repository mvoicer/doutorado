import pandas as pd
import random
import pickle
from tqdm import tqdm
import seaborn as sns
from matplotlib import pyplot as plt

from matrix_manipulation import create_subsample, merge_matrices, _create_subsample, _merge_matrices
from ml_models import fine_tunning
from correlation import *
from distance import Distance
from metrics import mse, rmse, r2, mape, accuracy


from mcdm import Gera_Pc_Mcdm, Mcdm_ranking
from consts import *



# Apenas checa se está fazendo testes no algoritmo principal ou não (neste caso, usa o dataframe completo)
def is_testing(check):
    if check == False:
        n_executions = 4                # número de vezes que vai rodar o algoritmo para tirar a média
        total_samples_per_rec = 51      # total de amostras a serem avaliadas a cada n_rec
        n_rec: int = 5                  # numero de amostras que são apresentadas ao decisor por vez
        CV: int = 5                     # number of cross-validation
        test_size: float = 0.2          # 80% train and 20% test
        top_n = n_rec                   # top n solutions
        max_sample = 120
    else:
        n_executions = 2
        total_samples_per_rec = 31
        n_rec: int = 5
        CV: int = 2
        test_size: float = 0.2
        top_n = n_rec
        max_sample = 51
    return n_executions, total_samples_per_rec, n_rec, CV, test_size, top_n, max_sample


# Load dataframe
def load_dataset(df, n_samples):
    df_var = pd.read_csv('data/'+str(df)+'_DEC.CSV', header=None)  # decision variables
    df_var = df_var.iloc[0:n_samples, :].round(5)
    df_obj = pd.read_csv('data/'+str(df)+'_OBJ.CSV', header=None)  # values in Pareto front
    df_obj = df_obj.iloc[0:n_samples, :].round(5)
    return df_var, df_obj


# Calculate similarities among the solutions
def calculate_similarities(df_obj, approach):
    # Calculate the distance among solutions
    if approach == 'Cosine':
        df_dist = Distance(df_obj).cosine()
    elif approach == 'Euclidean':
        df_dist = Distance(df_obj).euclidean()
    else:
        raise ValueError('Recommendation {} not implemented'.format(approach))
    return df_dist


# Initialize a dict for the results
def initialize_results():
    results = {}
    for alg in list_algs:
        results[alg] = {}
        for eng in recomm_engines:
            results[alg][eng] = {}
            for metr in list_metrics:
                results[alg][eng][metr] = []
    return results

# Generate the preferences
def generate_preferences(mcdm_method, df_obj):
    npop, nobj = df_obj.shape
    if mcdm_method == 'AHP':
        pc_matrix = Gera_Pc_Mcdm(df=df_obj).ahp_pc()
        rank_mcdm = Mcdm_ranking().ahp_ranking(pc_matrix=pc_matrix, weights=[.5, .5], nrow=npop, nobj=nobj)
    elif mcdm_method == 'Promethee':
        pc_matrix = Gera_Pc_Mcdm(df=df_obj, weights=[.5, .5], cb=['cost', 'cost']).promethee_ii_pc()
        rank_mcdm = Mcdm_ranking().promethee_ii_ranking(pc_matrix=pc_matrix, weights=None, nobj=nobj, nrow=npop)
    else:
        raise ValueError('Multicriteria method: {} not implemented'.format(mcdm_method))
    return pc_matrix, rank_mcdm


def calculate_new_ranking(mcdm_method, df_merged, weights, npop, nobj):
    if mcdm_method == 'AHP':
        rank_predicted = Mcdm_ranking().ahp_ranking(pc_matrix=df_merged, weights=None, nrow=npop, nobj=nobj)
    elif mcdm_method == 'Promethee':
        rank_predicted = Mcdm_ranking().promethee_ii_ranking(pc_matrix=df_merged, weights=None, nobj=nobj, nrow=npop)
    else:
        raise ValueError('Multicriteria method: {} not implemented'.format(mcdm_method))
    return rank_predicted
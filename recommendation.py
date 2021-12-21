import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm
import random
from matplotlib import pyplot as plt

from matrix_manipulation import create_subsample, merge_matrices, _create_subsample, _merge_matrices
from ml_models import fine_tunning
from correlation import norm_kendall, spearman_rho
from consts import *
from metrics import mse, rmse, r2, mape, accuracy
from pre_processing import load_dataset, is_testing, initialize_results, generate_preferences, calculate_similarities, \
    calculate_new_ranking
from clusterization import clusterization
np.seterr(divide='ignore', invalid='ignore')  # ignora erros de divisão por 0



def make_recommendation(type, most_similar, Q, tam):
    if type == 'most_similar':
        # Define Q and N-Q indexes
        for q in most_similar:
            if (len(Q) <= tam) & (q not in Q):
                Q.append(q)
            else:
                pass
    elif type == 'half_similar_half_dissimilar':
        # Half most similar
        for q in most_similar:
            if (len(Q) <= (tam - int(tam/2))) & (q not in Q):
                Q.append(q)
            else:
                # pass
                # Half most dissimilar
                most_dissimilar = most_similar.copy()
                most_dissimilar.reverse()
                for p in most_dissimilar:
                    if (len(Q) <= tam) & (p not in Q):
                        Q.append(p)
                    else:
                        continue
    else:
        raise ValueError("Recomendation {} not implemented".format(type))
    return Q


def initial_recommendation(type, indexes, df_obj, length):
    if type == 'aleatory':
        temp = indexes.copy()
        random.shuffle(temp)
        Q = temp[0:length]

    elif type == 'cluster':
        Q = clusterization(df=df_obj, type='kmeans', tam=length)
    N_Q = [x for x in indexes if x not in Q]

    return Q, N_Q


def run_recommender(dataframe, recomm_approach, mcdm_method, initial_recomm, weights, recomm_style, testing, date, print_pf):
    path_to_save = recomm_approach + '_' + mcdm_method + '_' + date
    # Load variables
    n_executions, total_samples_per_rec, n_rec, CV, test_size, top_n, max_sample = is_testing(testing)

    # Load dataframe
    df_var, df_obj = load_dataset(df=dataframe, n_samples=max_sample)
    npop, nvar = df_var.shape
    nobj = df_obj.shape[1]
    results = initialize_results()

    # Calculate the distances/similarities among the solutions
    df_dist = calculate_similarities(df_obj, recomm_approach)

    # Generate the preferences
    pc_matrix, rank_mcdm = generate_preferences(mcdm_method, df_obj)

    if print_pf is True:
        plt.scatter(df_obj.iloc[:, 0], df_obj.iloc[:, 1], color='grey', marker='o')  # available
        plt.scatter(df_obj.iloc[rank_mcdm[:n_rec], 0], df_obj.iloc[rank_mcdm[:n_rec], 1], color='green',
                    marker='v')  # primeiro elemento
        plt.scatter(df_obj.iloc[rank_mcdm[-n_rec:], 0], df_obj.iloc[rank_mcdm[-n_rec:], 1], color='black',
                    marker='*')  # ultimo elemento
        plt.ylim([-0.05, 2.2])
        plt.xlim([-0.05, 2.2])
        plt.show()

    # Initialize an arbitrary ranking to check convergence
    indexes = list(df_var.index)
    rank_aleatory = indexes.copy()
    random.shuffle(rank_aleatory)

    for alg in list_algs:
        for engine in recomm_engines:
            for exc in tqdm(range(n_executions)):
                print("\nEngine:", engine, "Algorithm:", alg, "Execution: ", exc)
                print("*" * 100)
                # Recommended solutions
                Q = []
                tam = n_rec
                # lista_aleatoria = indexes.copy()
                # random.shuffle(lista_aleatoria)
                for aux in range(n_rec, total_samples_per_rec, n_rec):
                    if (engine == 'aleatory') | (engine == 'personalized' and len(Q) == 0):
                        # Q = lista_aleatoria[0:tam]
                        # N_Q = [x for x in indexes if x not in Q]
                        Q, N_Q = initial_recommendation(type=initial_recomm,
                                                        indexes=indexes,
                                                        df_obj=df_obj,
                                                        length=tam)
                        # Plota soluções recomendadas
                        # plt.scatter(df_obj.iloc[:, 0], df_obj.iloc[:, 1], color='grey', marker='o')  # available
                        # plt.scatter(df_obj.iloc[Q, 0], df_obj.iloc[Q, 1], color='red', marker='v')
                        # plt.ylim([-0.05, 2.2])
                        # plt.xlim([-0.05, 2.2])
                        # plt.show()

                    elif engine == 'personalized':
                        # calculate most similar
                        most_similar = df_dist[rank_aleatory[0:tam]].sum(axis=1).sort_values(
                            ascending=True).index.to_list()
                        # Generate the list of recommendations
                        Q = make_recommendation(recomm_style, most_similar, Q, tam)
                        # Generate the remainings
                        N_Q = [x for x in indexes if x not in Q]

                    # Train and test set
                    X_train, y_train = create_subsample(df_var=df_var, df_pref=pc_matrix, nobj=nobj, index=Q)
                    X_test, y_test = create_subsample(df_var=df_var, df_pref=pc_matrix, nobj=nobj, index=N_Q)

                    # if mcdm_pairwise_method == 'Promethee':
                    # std = StandardScaler().fit(X_train)
                    # X_train = std.transform(X_train)
                    # X_test = std.transform(X_test)

                    # Fine tunning in the 1st execution and save the best model
                    if exc == 0:
                        # check if there are some trained model. if yes, read it.
                        if len(results[alg][engine]['tau']) <= int(total_samples_per_rec / n_rec):
                            try:
                                with open("tunned_models/tuned_model_" + alg + ".pkl", "rb") as fp:
                                    tuned_model = pickle.load(fp)
                            except:
                                pass
                        else:
                            pass
                        # train and tunning a model
                        tuned_model = fine_tunning(CV, X_train, y_train, algorithm=alg)
                        with open("tunned_models/tuned_" + alg + '_' + engine + "_" + path_to_save + ".pkl",
                                  'wb') as arq:
                            pickle.dump(tuned_model, arq)
                    else:
                        # if the execution is from 2nd on, just read the best model from the execution 0
                        with open("tunned_models/tuned_" + alg + '_' + engine + "_" + path_to_save + ".pkl",
                                  "rb") as fp:
                            tuned_model = pickle.load(fp)

                    # Model evaluation
                    y_pred = tuned_model.predict(X_test)
                    y_pred = pd.DataFrame(y_pred)
                    if mcdm_method == 'AHP':
                        y_pred = round(y_pred)

                    # Regularization 1: replace by 9 if the predicted value is greater than 9 (Saaty's scale max value)
                    y_pred[y_pred > 9] = 9

                    # Calculate metrics
                    results[alg][engine]['accuracy'].append(accuracy(y_pred, y_test))
                    results[alg][engine]['mse'].append(mse(y_pred, y_test))
                    results[alg][engine]['rmse'].append(rmse(y_pred, y_test))
                    results[alg][engine]['r2'].append(r2(y_pred, y_test))
                    results[alg][engine]['mape'].append(mape(y_pred, y_test))

                    # Merge the predictions of the df train and df test
                    df_merged = merge_matrices(N_Q, pc_matrix, y_pred)

                    # Calculate the predicted ranking
                    rank_predicted = calculate_new_ranking(mcdm_method, df_merged, weights=weights, npop=npop, nobj=nobj)

                    # plt.scatter(df_obj.iloc[:, 0], df_obj.iloc[:, 1], color='grey', marker='o')  # available
                    # plt.scatter(df_obj.iloc[rank_predicted[:aux], 0], df_obj.iloc[rank_predicted[:aux], 1], color='red',
                    #             marker='v')  # primeiro elemento
                    # plt.scatter(df_obj.iloc[Q[:aux], 0], df_obj.iloc[Q[:aux], 1], color='black', marker='*')  # ultimo elemento
                    # plt.ylim([-0.05, 2.2])
                    # plt.xlim([-0.05, 2.2])
                    # plt.show()

                    # Computing tau similarity
                    results[alg][engine]['tau'].append(norm_kendall(rank_aleatory, rank_predicted))
                    results[alg][engine]['rho'].append(spearman_rho(rank_aleatory, rank_predicted))

                    # Update the ranking
                    rank_aleatory = rank_predicted
                    tam += n_rec

    # Save results
    filename = open('results/results_' + path_to_save + ".pkl", 'wb')
    pickle.dump(results, filename)
    filename.close()

    return results


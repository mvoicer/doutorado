import pandas as pd
import numpy as np
import joblib
import random
import math
import time

from matrix_manipulation import create_subsample, merge_matrices
from ml_models import fine_tunning
from correlation import norm_kendall, spearman_rho
from metrics import *
from matplotlib import pyplot as plt
from pre_processing import load_dataset, initialize_results, generate_preferences, \
    calculate_similarities, calculate_mcdm_ranking
from recomm_manipulation import initial_recommendation, make_recommendation
from condorcet import condorcet
from visualization import Visualization

np.seterr(divide='ignore', invalid='ignore')  # ignora erros de divisão por 0


def run_recommender(dataframe, n_rec, mcdm_method, weights, cost_benefit, percent_random,
                    initial_recomm, similarity_measure, ml_method, date, plot_pareto_front,
                    plot_recommended_solutions, n_executions, total_samples_per_rec, CV):
    # How to save the results -- define name
    path_to_save = (date + '__' +
                    mcdm_method +
                    '__theta_' + str(percent_random) +
                    '__init_' + initial_recomm +
                    '__sim_' + similarity_measure +
                    '__ml_' + ml_method).lower()

    df_var, df_obj = load_dataset(df=dataframe)
    npop, nvar = df_var.shape
    nobj = df_obj.shape[1]
    results = initialize_results()
    # results_treino = initialize_results()

    # Calculate the distances/similarities among the solutions
    df_dist = calculate_similarities(df_obj, similarity_measure)

    # Generate the preferences
    pc_matrix, rank_mcdm = generate_preferences(mcdm_method, df_obj, weights=weights, cb=cost_benefit)

    if plot_pareto_front is True:
        Visualization.plot_pareto_front(df_obj)

    # Initialize an arbitrary ranking to check convergence
    indexes = list(df_var.index)

    # Initialize number of queries
    number_queries = 0
    accepted_error = 0.05

    for exc in (range(n_executions)):
        rank_aleatory = indexes.copy()
        rank_aleatory.reverse()
        print("*" * 100)
        print("\nExecution: ", exc)
        print("*" * 100)
        # Recommended solutions
        Q = []
        temp_error = 1
        iteracao = 0
        start_time = time.time()
        for aux in range(n_rec, total_samples_per_rec, n_rec):
            iteracao += 1
            print("*" * 50)
            print("Iteracao: ", iteracao)
            print("*" * 15)

            # 2nd iteration on
            if len(Q) != 0:
                # Calculate the distances/similarities among the solutions
                most_similar = df_dist.iloc[rank_predicted[0]].sort_values(ascending=True).index.to_list()
                print("Most similar: ", most_similar)
                # Generate the list of recommendations
                entrance = make_recommendation(most_similar, Q, n_rec, math.floor(n_rec * percent_random))

                # Get the preferences for the new alternatives (entrance) and for the last one in Q
                # (for the condorcet/regularization) and merge to the train and test set
                X_train_, y_train_ = create_subsample(df_var=df_var, df_pref=pc_matrix, nobj=nobj, index=entrance + Q)
                # X_train_cond, y_train_cond = condorcet(Q, entrance, pc_matrix, df_var, nobj)
                # X_train = pd.concat([X_train_, X_train_cond, X_train], axis=0, ignore_index=True)
                # y_train = pd.concat([y_train_, y_train_cond, y_train], axis=0, ignore_index=True)
                X_train = pd.concat([X_train_, X_train], axis=0, ignore_index=True)
                y_train = pd.concat([y_train_, y_train], axis=0, ignore_index=True)

                # Update list of recommendations
                Q = Q + entrance

                # Generate the remaining ones
                N_Q = [x for x in indexes if x not in Q]
                X_test, y_test = create_subsample(df_var=df_var, df_pref=pc_matrix, nobj=nobj, index=N_Q)

            # 1st iteration
            else:
                Q, N_Q = initial_recommendation(type_rec=initial_recomm, indexes=indexes, df_obj=df_obj, length=n_rec)
                # Train and test set
                X_train, y_train = create_subsample(df_var=df_var, df_pref=pc_matrix, nobj=nobj, index=Q)
                X_test, y_test = create_subsample(df_var=df_var, df_pref=pc_matrix, nobj=nobj, index=N_Q)

            if plot_recommended_solutions is True:
                Visualization.plot_recommended_solutions(df_obj, Q, rank_aleatory, rank_mcdm, n_rec)

            # Fine tunning
            if temp_error > accepted_error:
                start_fine_tunning = time.time()
                tuned_model = fine_tunning(CV, X_train, y_train, algorithm=ml_method)
                print("Time to fine tunning: %s seconds" % (time.time() - start_fine_tunning))

                start_fit_model = time.time()
                tuned_model.fit(X_train, y_train)

                joblib.dump(tuned_model, "experiments/varia_theta/pf1/tunned_models/" + path_to_save + '.gz')
            else:
                tuned_model = joblib.load("experiments/varia_theta/pf1/tunned_models/" + path_to_save + ".gz")

            # Model evaluation
            y_pred = tuned_model.predict(X_test)
            y_pred = pd.DataFrame(y_pred)

            # Regularization 1: replace by 9 if the predicted value is greater than 9 (Saaty's scale max value)
            y_pred[y_pred > 9] = 9
            y_pred[y_pred < -9] = -9

            # Visualization.scatter_predicted(nobj, y_test, y_pred)
            # Visualization.hist_residuals(nobj, y_test, y_pred)

            # Calculate metrics
            results['mse'].append(mse(y_pred=y_pred, y_true=y_test))
            results['rmse'].append(rmse(y_pred=y_pred, y_true=y_test))
            results['r2'].append(r2(y_pred=y_pred, y_true=y_test))
            results['mape'].append(mape(y_pred=y_pred, y_true=y_test))
            print("MSE: {}, RMSE: {}, R2: {}, MAPE: {}".format(mse(y_pred=y_pred, y_true=y_test),
                                                               rmse(y_pred=y_pred, y_true=y_test),
                                                               r2(y_pred=y_pred, y_true=y_test),
                                                               mape(y_pred=y_pred, y_true=y_test)))

            # Merge the predictions of train and test sets
            df_merged = merge_matrices(N_Q, pc_matrix, y_pred)

            # Calculate the predicted ranking
            rank_predicted = calculate_mcdm_ranking(mcdm_method, df_merged, weights=weights,
                                                    npop=npop, nobj=nobj, cb=cost_benefit)
            print('Rank predicted: \n', rank_predicted)
            print('Rank mcdm: \n', rank_mcdm)
            print('Rank aleatorio \n', rank_aleatory)

            # Computing tau between solutions
            temp_error = norm_kendall(rank_aleatory, rank_predicted)
            results['tau'].append(temp_error)
            results['rho'].append(spearman_rho(rank_aleatory, rank_predicted))
            results['mcdm'].append(norm_kendall(rank_mcdm, rank_predicted))
            print("Tau com ranking mcdm: {}, ranking aleatório: {}".format(norm_kendall(rank_mcdm, rank_predicted),
                                                                           temp_error))

            # Calculate number of queries
            number_queries += (n_rec * (n_rec - 1) / 2) * nobj
            print('Number of queries: ', number_queries)


            # Update the ranking
            rank_aleatory = rank_predicted

        # Save results
        joblib.dump(results, 'experiments/varia_theta/pf1/results/' + path_to_save + '.gz')
        print("Time to run execution", exc,": %s seconds" % (time.time() - start_time))

    return results

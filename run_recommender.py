import pandas as pd
import numpy as np
import joblib
import random
from matplotlib import pyplot as plt
import math

from matrix_manipulation import create_subsample, merge_matrices, _create_subsample, _merge_matrices
from ml_models import fine_tunning
from correlation import norm_kendall, spearman_rho
from params import *
from metrics import *
from pre_processing import load_dataset, initialize_results, generate_preferences, \
    calculate_similarities, calculate_new_ranking
from recomm_manipulation import initial_recommendation, make_recommendation
from condorcet import condorcet
from visualization import Visualization

np.seterr(divide='ignore', invalid='ignore')  # ignora erros de divisão por 0


def run_recommender(dataframe, n_rec, mcdm_method, weights, cost_benefit, percent_random,
                    initial_recomm, similarity_measure, ml_method, date, plot_pareto_front,
                    plot_recommended_solutions):
    # How to save the results
    # global y_train, X_train
    path_to_save = (mcdm_method + '_' + str(
        percent_random) + 'initrandom_sim' + similarity_measure + '_' + ml_method + '_' + date).lower()

    # Load variables
    n_executions, total_samples_per_rec, max_sample, CV = load_params()
    df_var, df_obj = load_dataset(df=dataframe, n_samples=max_sample)
    # print('')
    # print(df_var.to_string())
    npop, nvar = df_var.shape
    nobj = df_obj.shape[1]
    results = initialize_results()

    # Calculate the distances/similarities among the solutions
    df_dist = calculate_similarities(df_obj, similarity_measure)

    # Generate the preferences
    pc_matrix, rank_mcdm = generate_preferences(mcdm_method, df_obj, weights=weights, cb=cost_benefit)
    # print('')
    # print(pc_matrix.to_string())
    if plot_pareto_front is True:
        Visualization.plot_pareto_front(df_obj, rank_mcdm, n_rec)

    # Initialize an arbitrary ranking to check convergence
    indexes = list(df_var.index)
    rank_aleatory = indexes.copy()
    random.shuffle(rank_aleatory)

    for exc in (range(n_executions)):
        print("\nExecution: ", exc)
        print("*" * 100)
        # Recommended solutions
        Q = []
        temp_error = 1
        accepted_error = 0.05
        for aux in range(n_rec, total_samples_per_rec, n_rec):
            # 1st iteration?
            if len(Q) == 0:
                Q, N_Q = initial_recommendation(type_rec=initial_recomm, indexes=indexes, df_obj=df_obj, length=n_rec)
                # Train and test set
                X_train, y_train = _create_subsample(df_var=df_var, df_pref=pc_matrix, nobj=nobj, index=Q)
                X_test, y_test = _create_subsample(df_var=df_var, df_pref=pc_matrix, nobj=nobj, index=N_Q)
            else:
                # Calculate the distances/similarities among the solutions
                most_similar = df_dist.iloc[rank_aleatory[0]].sort_values(ascending=True).index.to_list()

                # Generate the list of recommendations
                entrance = make_recommendation(most_similar, Q, n_rec, math.floor(n_rec * percent_random))

                # Get the preferences for the new alternatives (entrance) and
                # for the last one in Q (for the condorcet/regularization) and
                # merge to the train and test set
                X_train_, y_train_ = _create_subsample(df_var=df_var, df_pref=pc_matrix, nobj=nobj, index=entrance + Q)
                X_train_cond, y_train_cond = condorcet(Q, entrance, pc_matrix, df_var, nobj)
                X_train = pd.concat([X_train_, X_train_cond, X_train], axis=0, ignore_index=True)
                y_train = pd.concat([y_train_, y_train_cond, y_train], axis=0, ignore_index=True)

                # Update list of recommendations
                Q = Q + entrance

                # Generate the remaining ones
                N_Q = [x for x in indexes if x not in Q]
                X_test, y_test = _create_subsample(df_var=df_var, df_pref=pc_matrix, nobj=nobj, index=N_Q)
            # print('Q', Q)
            if plot_recommended_solutions is True:
                Visualization.plot_recommended_solutions(df_obj, Q, rank_aleatory, n_rec)

            # Fine tunning in the 1st execution, otherwise load the model saved
            if temp_error > accepted_error:
                # print('Treinou e o erro atual é: ', temp_error)
                tuned_model = fine_tunning(CV, X_train, y_train, algorithm=ml_method)
                tuned_model.fit(X_train, y_train)
                joblib.dump(tuned_model, "tunned_models/" + path_to_save + '.gz')
            else:
                # print('Testou. Erro: ', temp_error)
                tuned_model = joblib.load("tunned_models/" + path_to_save + ".gz")

            # Model evaluation
            y_pred = tuned_model.predict(X_test)
            y_pred = pd.DataFrame(y_pred)
            if mcdm_method == 'AHP':
                # Regularization 1: replace by 9 if the predicted value is greater than 9 (Saaty's scale max value)
                y_pred[y_pred > 9] = 9
                y_pred[y_pred < .1] = .1

            # Visualization.scatter_predicted(nobj, y_test, y_pred)
            # Visualization.hist_residuals(n obj, y_test, y_pred)

            # Calculate metrics
            results['mse'].append(mse(y_pred=y_pred, y_true=y_test))
            results['rmse'].append(rmse(y_pred=y_pred, y_true=y_test))
            results['r2'].append(r2(y_pred=y_pred, y_true=y_test))
            results['mape'].append(mape(y_pred=y_pred, y_true=y_test))

            # Merge the predictions of the df train and df test
            df_merged = _merge_matrices(N_Q, pc_matrix, y_pred)

            # Calculate the predicted ranking
            rank_predicted = calculate_new_ranking(mcdm_method, df_merged, weights=weights, npop=npop, nobj=nobj)

            # Computing tau similarity
            temp_error = norm_kendall(rank_aleatory, rank_predicted)
            results['tau'].append(temp_error)
            results['rho'].append(spearman_rho(rank_aleatory, rank_predicted))
            results['mcdm'].append(norm_kendall(rank_aleatory, rank_mcdm))

            # Update the ranking
            rank_aleatory = rank_predicted

    # Save results
    joblib.dump(results, 'results/' + path_to_save + '.gz')

    return results

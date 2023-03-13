import pandas as pd
import joblib
import math
import time
from matrix_manipulation import create_subsample, merge_matrices
from ml_models import fine_tunning
from metrics import *
from distance import calculate_similarities
from pre_processing import load_dataset, initialize_results, generate_preferences, calculate_mcdm_ranking
from recomm_manipulation import initial_recommendation, make_recommendation
from visualization import Visualization
from ahp_weights import calculate_ahp_weights
from params import *

np.seterr(divide='ignore', invalid='ignore')  # ignora erros de divisão por 0


def run_recommender(dataframe, n_rec, mcdm_method, weights, cost_benefit, theta, iota, zeta,
                    ml_method, date, n_executions, total_samples_per_rec,
                    plot_pareto_front=False, plot_recommended_solutions=False,
                    scatter_predicted=False, hist_residuals=False):

    # How to save the results -- define name
    path_to_save = (date + '__' + mcdm_method + '__theta_' + str(theta) + '__init_' + iota +
                    '__sim_' + zeta + '__ml_' + ml_method).lower()

    df_var, df_obj = load_dataset(df=dataframe)
    npop, nvar = df_var.shape
    nobj = df_obj.shape[1]
    results = initialize_results()

    # Calculate the distances/similarities among the solutions
    df_dist = calculate_similarities(df_obj, zeta)

    # AHP weights
    if mcdm_method == 'AHP' and weights is None:
        weights = calculate_ahp_weights(df_obj)

    # Generate the preferences
    pc_matrix, rank_mcdm = generate_preferences(mcdm_method, df_obj, weights=weights, cb=cost_benefit)

    if plot_pareto_front:
        Visualization.plot_pareto_front(df_obj)

    # Initialize an arbitrary ranking to check convergence
    indexes = list(df_var.index)

    for exc in (range(n_executions)):
        rank_aleatory = indexes.copy()
        rank_aleatory.reverse()
        print("*" * 100)
        print("\nExecution: ", exc)
        print("*" * 100)
        # Recommended solutions
        Q = []
        temp_error = 1.0
        number_queries = 0
        t = 0
        start_time = time.time()
        for _ in range(n_rec, total_samples_per_rec, n_rec):
            t += 1
            print("*" * 50)
            print("Iteration ", t)
            print("*" * 50)

            # 2nd iteration on
            if len(Q) != 0:
                # Calculate the distances/similarities among the solutions
                most_similar = df_dist.iloc[rank_predicted[0]].sort_values(ascending=True).index.to_list()
                print("Most similar: ", most_similar)
                entrance = make_recommendation(most_similar, Q, n_rec, math.floor(n_rec * theta))

                X_train_, y_train_ = create_subsample(df_var=df_var, df_pref=pc_matrix, nobj=nobj, index=entrance + Q)
                # X_train_cond, y_train_cond = condorcet(Q, entrance, pc_matrix, df_var, nobj)
                # X_train = pd.concat([X_train_, X_train_cond, X_train], axis=0, ignore_index=True)
                # y_train = pd.concat([y_train_, y_train_cond, y_train], axis=0, ignore_index=True)
                X_train = pd.concat([X_train_, X_train], axis=0, ignore_index=True)
                y_train = pd.concat([y_train_, y_train], axis=0, ignore_index=True)

                # Update list of recommendations
                Q = Q + entrance
                N_Q = [x for x in indexes if x not in Q]
                X_test, y_test = create_subsample(df_var=df_var, df_pref=pc_matrix, nobj=nobj, index=N_Q)

            # 1st iteration
            else:
                Q, N_Q = initial_recommendation(type_rec=iota, indexes=indexes, df_obj=df_obj, length=n_rec)
                X_train, y_train = create_subsample(df_var=df_var, df_pref=pc_matrix, nobj=nobj, index=Q)
                X_test, y_test = create_subsample(df_var=df_var, df_pref=pc_matrix, nobj=nobj, index=N_Q)

            if plot_recommended_solutions:
                Visualization.plot_recommended_solutions(df_obj, Q, rank_aleatory, rank_mcdm, n_rec)

            # Fine tunning
            if temp_error > epsilon:
            # if len(Q) == n_rec:
                start_fine_tunning = time.time()
                tuned_model = fine_tunning(X_train, y_train, algorithm=ml_method)
                print("Time to fine tunning: %s seconds" % (time.time() - start_fine_tunning))
                tuned_model.fit(X_train, y_train)
                joblib.dump(tuned_model, "experiments/tanabe_ishibuchi/re243/tunned_models/" + path_to_save + '__exec_' + str(exc) +'.gz')
            else:
                tuned_model = joblib.load("experiments/tanabe_ishibuchi/re243/tunned_models/" + path_to_save + '__exec_' + str(exc) +'.gz')

            # Model evaluation
            y_pred = tuned_model.predict(X_test)
            y_pred = pd.DataFrame(y_pred)

            # Regularization 1: replace by 9 if the predicted value is greater than 9 (Saaty's scale max value)
            y_pred[y_pred > 9] = 9
            y_pred[y_pred < -9] = -9

            if scatter_predicted:
                Visualization.scatter_predicted(nobj, y_test, y_pred)
            if hist_residuals:
                Visualization.hist_residuals(nobj, y_test, y_pred)

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

            # Compute metrics
            temp_error = norm_kendall(rank_aleatory, rank_predicted)
            results['tau'].append(temp_error)
            results['mcdm'].append(norm_kendall(rank_mcdm, rank_predicted))
            results['pos'].append(position_match_rate(rank_mcdm, rank_predicted))
            results['dbs'].append(dbs(rank_mcdm, rank_predicted))
            print("Tau com ranking mcdm: {}, ranking aleatório: {}".format(norm_kendall(rank_mcdm, rank_predicted),
                                                                           temp_error))

            # Calculate NQ
            number_queries += (n_rec * (n_rec - 1) / 2) * nobj
            results['nq'].append(number_queries)

            # Update the ranking
            rank_aleatory = rank_predicted

        # Save results
        joblib.dump(results, 'experiments/tanabe_ishibuchi/re243/results/' + path_to_save + '__exec_' + str(exc) +'.gz')
        print("Time to run execution", exc,": %s seconds" % (time.time() - start_time))

    return results

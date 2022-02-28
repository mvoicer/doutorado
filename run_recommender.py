import pandas as pd
import numpy as np
import pickle
import joblib
import random
from tqdm import tqdm
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

np.seterr(divide='ignore', invalid='ignore')  # ignora erros de divisão por 0


def run_recommender(dataframe, n_rec, mcdm_method, weights, cost_benefit, percent_random,
                    initial_recomm, similarity_measure, ml_method, date, plot_pareto_front,
                    plot_recommended_solutions):
    # How to save the results
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
        if df_obj.shape[1] == 2:
            plt.scatter(df_obj.iloc[:, 0], df_obj.iloc[:, 1], color='grey', marker='o', label='Available')
            plt.scatter(df_obj.iloc[rank_mcdm[:n_rec], 0], df_obj.iloc[rank_mcdm[:n_rec], 1],
                        color='green', marker='v', label='Best mcdm ranked')
            plt.scatter(df_obj.iloc[rank_mcdm[-n_rec:], 0], df_obj.iloc[rank_mcdm[-n_rec:], 1],
                        color='black', marker='*', label='Worst mcdm ranked')
            plt.ylim(ymin=0)
            plt.xlim(xmin=0)
            plt.legend(loc='best')
            plt.title('Best MCDM ranked solutions in the PF')
            plt.show()
        elif df_obj.shape[1] == 3:
            ax = plt.axes(projection='3d')
            ax.scatter3D(df_obj.iloc[:, 0], df_obj.iloc[:, 1], df_obj.iloc[:, 2],
                         color='grey', marker='o', label='Available')
            ax.scatter3D(df_obj.iloc[rank_mcdm[:n_rec], 0], df_obj.iloc[rank_mcdm[:n_rec], 1],
                         df_obj.iloc[rank_mcdm[:n_rec], 2], color='green', marker='v', label='Best ranked')
            ax.scatter3D(df_obj.iloc[rank_mcdm[-n_rec:], 0], df_obj.iloc[rank_mcdm[-n_rec:], 1],
                         df_obj.iloc[rank_mcdm[-n_rec:], 2], color='black', marker='*', label='Worst ranked')
            ax.view_init(30, 30)
            ax.set_xlabel('Obj 1')
            ax.set_ylabel('Obj 2')
            ax.set_zlabel('Obj 3')
            ax.legend(loc='best')
            ax.set_title('Best MCDM ranked solutions in the PF')
            plt.show()
        else:
            # plot in parallel coordinates
            import plotly.express as px

            fig = px.parallel_coordinates(df_obj,
                                          dimensions=df_obj.columns,
                                          labels={k: v for k, v in
                                                  enumerate(['Obj ' + str(i) for i in df_obj.columns])})

            fig.show()

    # Initialize an arbitrary ranking to check convergence
    indexes = list(df_var.index)
    rank_aleatory = indexes.copy()
    random.shuffle(rank_aleatory)

    for exc in (range(n_executions)):
        print("\nExecution: ", exc)
        print("*" * 100)
        # Recommended solutions
        Q = []
        for aux in range(n_rec, total_samples_per_rec, n_rec):
            # 1st iteration?
            if len(Q) == 0:
                Q, N_Q = initial_recommendation(type_rec=initial_recomm, indexes=indexes, df_obj=df_obj, length=n_rec)
                # Train and test set
                X_train, y_train = _create_subsample(df_var=df_var, df_pref=pc_matrix, nobj=nobj, index=Q)
                X_test, y_test = _create_subsample(df_var=df_var, df_pref=pc_matrix, nobj=nobj, index=N_Q)
            else:
                # Calculate the distances/similarities among the solutions
                # most_similar = df_dist[rank_aleatory[0:tam]].sum(axis=1).sort_values(ascending=True).index.to_list()
                most_similar = df_dist.iloc[rank_aleatory[0]].sort_values(ascending=True).index.to_list()
                # Qtd random solutions to be added per iteration
                qtd_to_add = math.floor(n_rec * percent_random)
                # Generate the list of recommendations
                entrance = make_recommendation(most_similar, Q, n_rec, qtd_to_add)

                # Get the preferences for the new alternatives (entrance) and
                # for the last one in Q (for the condorcet/regularization) and
                # merge to the train and test set
                X_train_, y_train_ = _create_subsample(df_var=df_var, df_pref=pc_matrix, nobj=nobj,  index=entrance+Q)

                # Condorcet
                # Check the preferences between the last solution in Q and the others in entrance
                # and infer new possible preferences to increase the number of samples in the training
                # Ex: if Ai > Aj = 9 and Aj > An = 2, then Ai (will possible be) > An = 7.
                X_train_cond, y_train_cond = condorcet(Q, entrance, pc_matrix, df_var, nobj)

                X_train = pd.concat([X_train_, X_train_cond, X_train], axis=0, ignore_index=True)
                y_train = pd.concat([y_train_, y_train_cond, y_train], axis=0, ignore_index=True)

                # Update Q
                Q = Q + entrance

                # Generate the remaining ones
                N_Q = [x for x in indexes if x not in Q]
                X_test, y_test = _create_subsample(df_var=df_var, df_pref=pc_matrix, nobj=nobj, index=N_Q)

            # Fine tunning in the 1st execution and save the best model
            if exc == 0:
                # check if there are some trained model. if yes, read it.
                if len(results['tau']) <= int(total_samples_per_rec / n_rec):
                    try:
                        tuned_model = joblib.load("tunned_models/" + path_to_save + ".gz")
                    except:
                        pass
                else:
                    pass
                # train and tunning a model
                tuned_model = fine_tunning(CV, X_train, y_train, algorithm=ml_method)
                joblib.dump(tuned_model, "tunned_models/" + path_to_save + '.gz')
            else:
                # if 2nd interation on, just read the best model from the execution 0
                tuned_model = joblib.load("tunned_models/" + path_to_save + '.gz')

            # Model evaluation
            y_pred = tuned_model.predict(X_test)
            y_pred = pd.DataFrame(y_pred)
            if mcdm_method == 'AHP':
                y_pred = y_pred.astype(int)
                # Regularization 1: replace by 9 if the predicted value is greater than 9 (Saaty's scale max value)
                y_pred[y_pred > 9] = 9

            # Calculate metrics
            results['mse'].append(mse(y_pred=y_pred, y_true=y_test))
            results['rmse'].append(rmse(y_pred=y_pred, y_true=y_test))
            results['r2'].append(r2(y_pred=y_pred, y_true=y_test))
            results['mape'].append(mape(y_pred=y_pred, y_true=y_test))

            # Merge the predictions of the df train and df test
            df_merged = _merge_matrices(N_Q, pc_matrix, y_pred)

            # Calculate the predicted ranking
            rank_predicted = calculate_new_ranking(mcdm_method, df_merged, weights=weights, npop=npop, nobj=nobj)

            if plot_recommended_solutions is True:
                plt.scatter(df_obj.iloc[:, 0], df_obj.iloc[:, 1], color='grey', marker='o', label='Available')
                plt.scatter(df_obj.iloc[rank_predicted[:aux], 0], df_obj.iloc[rank_predicted[:aux], 1],
                            color='red', marker='v', label='Best rank predicted')
                plt.scatter(df_obj.iloc[Q[:aux], 0], df_obj.iloc[Q[:aux], 1],
                            color='black', marker='*', label='Recommended')
                plt.ylim(ymin=0)
                plt.xlim(xmin=0)
                plt.title('Recommended solutions')
                plt.legend(loc='best')
                # plt.ylim([-0.05, 2.2])
                # plt.xlim([-0.05, 2.2])
                plt.show()

            # Computing tau similarity
            results['tau'].append(norm_kendall(rank_aleatory, rank_predicted))
            results['rho'].append(spearman_rho(rank_aleatory, rank_predicted))

            # Update the ranking
            rank_aleatory = rank_predicted

    # Save results
    joblib.dump(results, 'results/' + path_to_save + '.gz')

    return results

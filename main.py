import pandas as pd
import random
import pickle
from tqdm import tqdm
import seaborn as sns
from matplotlib import pyplot as plt

from data_preparation import create_subsample, merge_matrices, _create_subsample, _merge_matrices
from fine_tunning import fine_tunning
from correlation import *
from consts import *
from distance import Distance
from metrics import mse, rmse, r2, mape, accuracy
from mcdm import Gera_Pc_Mcdm, Mcdm_ranking

# NSGAIII_GPD03_M2_DEC.CSV -- 2 objetivos, convexa e multimodal
# NSGAIII_GPD04_M3_DEC.CSV -- 3 objetivos
df_var = pd.read_csv("data/NSGAIII_GPD04_M2_DEC.CSV", header=None)  # decision variables
df_var = df_var.iloc[0:max_sample, :].round(5)
df_obj = pd.read_csv('data/NSGAIII_GPD04_M2_OBJ.CSV', header=None)  # values in Pareto front
df_obj = df_obj.iloc[0:max_sample, :].round(5)

npop, nvar = df_var.shape
nobj = df_obj.shape[1]

# Calculate the distance among solutions
if not True:
    df_dist = Distance(df_obj).euclidean()
    wg = 'euclidean'
if True:
    df_dist = Distance(df_obj).cosine()
    wg = 'cosine'

# Generate the preferences
if True:
    pc_matrix = Gera_Pc_Mcdm(df=df_obj, weights=[.5, .5], cb=['cost', 'cost']).ahp_pc()
if not True:
    pc_matrix = Gera_Pc_Mcdm(df=df_obj, weights=[.5, .5], cb=['cost', 'cost']).promethee_ii_pc()

# plt.scatter(df_obj.loc[:, 0], df_obj.loc[:, 1], color='grey', marker='o')  # available
# plt.scatter(df_obj.loc[rank_mcdm[:10], 0], df_obj.loc[rank_mcdm[:10], 1], color='blue', marker='v')  # primeiro elemento
# # plt.scatter(df_obj.loc[rank_mcdm[-1:], 0], df_obj.loc[rank_mcdm[-1:], 1], color='red', marker='v')  # ultimo elemento
# plt.show()

# Generate the index to be evaluated
alternatives = list(df_var.index)

# Initialize an arbitrary ranking to check convergence
rank_aleatory = alternatives.copy()
random.shuffle(rank_aleatory)

for r in recomm_engines:
    for exc in tqdm(range(n_executions)):
        print("\nEngine/execution: ", r, exc)
        # Recommended solutions
        Q = []
        lista_aleatoria = alternatives.copy()
        random.shuffle(lista_aleatoria)
        for aux in range(n_rec, total_samples_per_rec, n_rec):
            if r == 'aleatory' or (r == 'euclidean' and len(Q) == 0):
                Q = lista_aleatoria[0:aux]
                N_Q = [x for x in alternatives if x not in Q]
            elif r == 'euclidean':
                # calculate most similar
                most_similar = df_dist[rank_aleatory[0:aux]].sum(axis=1).sort_values(ascending=True)[
                               0:n_rec * 2].index.to_list()

                # Define Q and N-Q indexes
                temp = len(Q) + n_rec
                for q in most_similar:
                    if (len(Q) < temp) & (q not in Q):
                        Q.append(q)
                    else:
                        continue
                N_Q = [x for x in alternatives if x not in Q]

            # Train
            df_Q = create_subsample(df_var=df_var, df_pref=pc_matrix, nobj=nobj, index=Q)
            X_train = df_Q.iloc[:, :-nobj]  # to predict
            y_train = df_Q.iloc[:, -nobj:]  # real targets
            # Test
            df_N_Q = create_subsample(df_var=df_var, df_pref=pc_matrix, nobj=nobj, index=N_Q)
            X_test = df_N_Q.iloc[:, :-nobj]  # to predict
            y_test = df_N_Q.iloc[:, -nobj:]  # real targets

            # Load trained model
            with open("models/tuned_model_random.pkl", "rb") as fp:
                tuned_model = pickle.load(fp)

            # # Fine tunning and save best model
            # if len(results[r]['tau']) <= int(total_samples_per_rec / n_rec):
            #     tuned_model = fine_tunning(CV, X_train, y_train)
            #     with open("tuned_model_"+r+".pkl", 'wb') as arq:
            #         pickle.dump(tuned_model, arq)
            # # Just load saved model
            # else:
            #     with open("tuned_model_"+r+".pkl", "rb") as fp:
            #         tuned_model = pickle.load(fp)

            # Model training
            tuned_model.fit(X_train, y_train)

            # Model evaluation
            y_pred = tuned_model.predict(X_test)
            y_pred = pd.DataFrame(y_pred)

            results[r]['accuracy'].append(accuracy(y_pred, y_test))
            results[r]['mse'].append(mse(y_pred, y_test))
            results[r]['rmse'].append(rmse(y_pred, y_test))
            results[r]['r2'].append(r2(y_pred, y_test))
            results[r]['mape'].append(mape(y_pred, y_test))

            # Merge the predictions of the df train and df test
            df_merged = merge_matrices(N_Q, pc_matrix, y_pred)

            # Employ AHP in the predicted (mixed with preferences) dataset
            rank_predicted = Mcdm_ranking(df_merged).ahp_ranking()

            # Computing tau similarity
            results[r]['tau'].append(norm_kendall(rank_aleatory, rank_predicted))
            results[r]['rho'].append(spearman_rho(rank_aleatory, rank_predicted))

            # Update the ranking
            rank_aleatory = rank_predicted

euc = pd.DataFrame(results['euclidean']['tau'])
euc['recommendation'] = wg
euc['iteracao'] = list(range(0, int(total_samples_per_rec / n_rec))) * n_executions

ale = pd.DataFrame(results['aleatory']['tau'])
ale['recommendation'] = 'aleatory'
ale['iteracao'] = list(range(0, int(total_samples_per_rec / n_rec))) * n_executions

data = pd.concat([euc, ale])

sns.boxplot(data=data, hue='recommendation', x='iteracao', y=0)
plt.xlabel("Iteration")
plt.ylabel("Similarity")
plt.ylim(bottom=0, top=1)
plt.axhline(y=.05, ls=':', color='red')

plt.show()

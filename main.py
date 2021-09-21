import pandas as pd
import random
import pickle
from tqdm import tqdm
import seaborn as sns
from matplotlib import pyplot as plt

from preferences import notas_pref
from ahp import ahp
from data_preparation import create_subsample
from fine_tunning import fine_tunning
from data_preparation import merge_matrices
from similarity import Similarity
from consts import *
from distance import Distance
from metrics import mse, rmse, r2, mape, accuracy

# NSGAIII_GPD03_M2_DEC.CSV -- 2 objetivos, convexa e multimodal
# NSGAIII_GPD04_M3_DEC.CSV -- 3 objetivos
df_var = pd.read_csv("data/NSGAIII_GPD04_M2_DEC.CSV", header=None)  # decision variables
df_var = df_var.iloc[0:105, :].round(5)
df_obj = pd.read_csv('data/NSGAIII_GPD04_M2_OBJ.CSV', header=None)  # values in Pareto front
df_obj = df_obj.iloc[0:105, :].round(5)

npop, nvar = df_var.shape
nobj = df_obj.shape[1]

# Calculate the similarities
df_dist = Distance(df_obj).rec_euclidean()

# Generate the preferences
df_obj = df_obj.to_numpy()
df_pref = notas_pref(df_obj)
df_obj = pd.DataFrame(df_obj)

# Apply the classical AHP to get the ranking
rank_ahp = ahp(df_pref).index

# Generate the index to be evaluated
index = list(df_var.index)
# Aleatory ranking
aleatory = index.copy()
random.shuffle(aleatory)
rank_aleatory = ahp(df_var.loc[aleatory]).index

for _ in range(n_executions):
    for r in tqdm(recomm_engines):
        Q = []
        random.shuffle(aleatory)
        for aux in range(N_REC, TOTAL, N_REC):
            if r == 'aleatory' or (r == 'euclidean' and len(Q) == 0):
                Q = aleatory[0:aux]
                N_Q = [x for x in index if x not in Q]
            elif r == 'euclidean':
                # recommend most similar (based on euclidean distance)
                most_similar = df_dist[rank_aleatory[0:aux]].sum(axis=1).sort_values(ascending=True)[0:N_REC].index.to_list()
                most_similar = most_similar[:N_REC]

                # Define Q and N-Q indexes
                Q = list(Q)
                for q in most_similar:
                    Q.append(q)
                N_Q = [x for x in index if x not in Q]

            # Train
            df_Q = create_subsample(df_var=df_var, df_pref=df_pref, nobj=nobj, index=Q)
            X_train = df_Q.iloc[:, :-nobj]  # to predict
            y_train = df_Q.iloc[:, -nobj:]  # real targets
            # Test
            df_N_Q = create_subsample(df_var=df_var, df_pref=df_pref, nobj=nobj, index=N_Q)
            X_test = df_N_Q.iloc[:, :-nobj]  # to predict
            y_test = df_N_Q.iloc[:, -nobj:]  # real targets

            # Load trained model
            with open("tuned_model_random.pkl", "rb") as fp:
                tuned_model = pickle.load(fp)

            # # Fine tunning and save best model
            # if len(results[r]['tau']) <= int(TOTAL / N_REC):
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
            df_merged = merge_matrices(N_Q, df_pref, y_pred, nobj, npop)

            # Employ AHP in the predicted (mixed with preferences) dataset
            rank_predicted = ahp(df_merged).index

            # Computing tau similarity
            results[r]['tau'].append(Similarity(rank_aleatory, rank_predicted).norm_kendall())

            # Plot pareto front and recommendations
            df_obj = pd.DataFrame(df_obj)
            plt.scatter(df_obj.loc[:, 0], df_obj.loc[:, 1], color='grey', marker='o')  # available
            plt.scatter(df_obj.loc[Q[0:aux], 0], df_obj.loc[Q[0:aux], 1], color='orange', marker='*')  # recommended
            plt.scatter(df_obj.loc[rank_predicted[0:aux], 0], df_obj.loc[rank_predicted[0:aux], 1], color='red', marker='^')  # top ranked
            plt.scatter(df_obj.loc[rank_ahp[0:aux], 0], df_obj.loc[rank_ahp[0:aux], 1], color='blue', marker='v')  # ahp
            plt.legend(["Available", 'Recommended', "Top ranked", 'AHP'])
            plt.show()

            # Update the ranking
            rank_aleatory = rank_predicted

euc = pd.DataFrame(results['euclidean']['tau'])
euc['recomm'] = 'euclidean'
euc['iteracao'] = list(range(0, int(TOTAL / N_REC))) * n_executions

ale = pd.DataFrame(results['aleatory']['tau'])
ale['recomm'] = 'aleatory'
ale['iteracao'] = list(range(0, int(TOTAL / N_REC))) * n_executions

data = pd.concat([euc, ale])

sns.boxplot(data=data, hue='recomm', x='iteracao', y=0)
plt.xlabel("Iteration")
plt.ylabel("Similarity")
plt.ylim(bottom=0, top=1)
plt.axhline(y=.05, ls=':', color='red')

plt.show()

import pandas as pd
import random
import pickle
from tqdm import tqdm
import seaborn as sns
from matplotlib import pyplot as plt

from data_preparation import create_subsample, merge_matrices, _create_subsample, _merge_matrices
from ml_models import fine_tunning
from correlation import *
from consts import *
from distance import Distance
from metrics import mse, rmse, r2, mape, accuracy
from mcdm import Gera_Pc_Mcdm, Mcdm_ranking
from sklearn.preprocessing import StandardScaler, Normalizer
np.seterr(divide='ignore', invalid='ignore')    # ignora erros de divisão por 0
from sklearn.linear_model import Lasso, LassoCV, MultiTaskLassoCV


# NSGAIII_GPD03_M2_DEC.CSV -- 2 objetivos, convexa e multimodal
# NSGAIII_GPD04_M3_DEC.CSV -- 3 objetivos
# NSGAIII_GPD04_M2_DEC.CSV -- 2 objetivos, concava
df_var = pd.read_csv("data/NSGAIII_GPD04_M2_DEC.CSV", header=None)  # decision variables
df_var = df_var.iloc[0:max_sample, :].round(5)
df_obj = pd.read_csv('data/NSGAIII_GPD04_M2_OBJ.CSV', header=None)  # values in Pareto front
df_obj = df_obj.iloc[0:max_sample, :].round(5)

npop, nvar = df_var.shape
nobj = df_obj.shape[1]

tipo_recomendacao = 'Euclidean'     # Cosine, Euclidean
mcdm_pairwise_method = 'AHP'        # AHP, Promethee
r_pers = 'half_similar_half_dissimilar'             # most_similar, half_similar_half_dissimilar
results = initialize_results()

# Calculate the distance among solutions
if tipo_recomendacao == 'Cosine':
    df_dist = Distance(df_obj).cosine()
elif tipo_recomendacao == 'Euclidean':
    df_dist = Distance(df_obj).euclidean()

# Generate the preferences
if mcdm_pairwise_method == 'AHP':
    pc_matrix = Gera_Pc_Mcdm(df=df_obj).ahp_pc()
    rank_mcdm = Mcdm_ranking().ahp_ranking(pc_matrix=pc_matrix, weights=[.5, .5], nrow=npop, nobj=nobj)
elif mcdm_pairwise_method == 'Promethee':
    pc_matrix = Gera_Pc_Mcdm(df=df_obj, weights=[.5, .5], cb=['cost', 'cost']).promethee_ii_pc()
    rank_mcdm = Mcdm_ranking().promethee_ii_ranking(pc_matrix=pc_matrix, weights=None, nobj=nobj, nrow=npop)

# plt.scatter(df_obj.iloc[:, 0], df_obj.iloc[:, 1], color='grey', marker='o')  # available
# plt.scatter(df_obj.iloc[rank_mcdm[:2], 0], df_obj.iloc[rank_mcdm[:2], 1], color='green', marker='v')  # primeiro elemento
# plt.scatter(df_obj.iloc[rank_mcdm[-1:], 0], df_obj.iloc[rank_mcdm[-1:], 1], color='black', marker='*')  # ultimo elemento
# plt.ylim([-0.05, 2.2])
# plt.xlim([-0.05, 2.2])
# plt.show()

# Generate the index to be evaluated
alternatives = list(df_var.index)

# Initialize an arbitrary ranking to check convergence
rank_aleatory = alternatives.copy()
random.shuffle(rank_aleatory)

for alg in list_algs:
    for engine in recomm_engines:
        for exc in tqdm(range(n_executions)):
            print("\nEngine:", engine, "Algorithm:", alg, "Execution: ",exc)
            print("*"*100)
            # Recommended solutions
            Q = []
            tam = n_rec
            lista_aleatoria = alternatives.copy()
            random.shuffle(lista_aleatoria)
            for aux in range(n_rec, total_samples_per_rec, n_rec):
                if (engine == 'aleatory') | (engine == 'personalized' and len(Q) == 0):
                    Q = lista_aleatoria[0:tam]
                    N_Q = [x for x in alternatives if x not in Q]
                elif engine == 'personalized':
                    # calculate most similar
                    most_similar = df_dist[rank_aleatory[0:tam]].sum(axis=1).sort_values(ascending=True).index.to_list()

                    if r_pers == 'most_similar':
                        # Define Q and N-Q indexes
                        for q in most_similar:
                            if (len(Q) <= tam) & (q not in Q):
                                Q.append(q)
                            else:
                                pass
                    elif r_pers == 'half_similar_half_dissimilar':
                        # Half most similar
                        for q in most_similar:
                            if (len(Q) <= (tam - 2)) & (q not in Q):
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
                        N_Q = [x for x in alternatives if x not in Q]

                # Train
                df_Q = create_subsample(df_var=df_var, df_pref=pc_matrix, nobj=nobj, index=Q)
                X_train = df_Q.iloc[:, :-nobj]  # to predict
                y_train = df_Q.iloc[:, -nobj:]  # real targets
                # Test
                df_N_Q = create_subsample(df_var=df_var, df_pref=pc_matrix, nobj=nobj, index=N_Q)
                X_test = df_N_Q.iloc[:, :-nobj]  # to predict
                y_test = df_N_Q.iloc[:, -nobj:]  # real targets

                # if mcdm_pairwise_method == 'Promethee':
                # std = StandardScaler().fit(X_train)
                # X_train = std.transform(X_train)
                # X_test = std.transform(X_test)

                # Load trained model
                # with open("models/tuned_model_random.pkl", "rb") as fp:
                #     tuned_model = pickle.load(fp)
                # tuned_model = MultiTaskLassoCV(n_alphas=100, normalize=True, random_state=1, max_iter=60000000, cv=5, n_jobs=-1)

                # Fine tunning in the 1st execution and save the best model
                if exc == 0:
                    # check if there are some trained model. if yes, read it.
                    if len(results[alg][engine]['tau']) <= int(total_samples_per_rec / n_rec):
                        try:
                            with open("models/tuned_model_" + engine + "_" + alg + ".pkl", "rb") as fp:
                                tuned_model = pickle.load(fp)
                        except:
                            pass
                    else:
                        pass
                    # train and tunning a model
                    tuned_model = fine_tunning(CV, X_train, y_train, algorithm=alg)
                    with open("models/tuned_model_"+engine+"_"+alg+'_'+date+".pkl", 'wb') as arq:
                        pickle.dump(tuned_model, arq)
                    print("Saved model: \n", tuned_model)
                else:
                    # if the execution is from 2nd on, just read the best model from the execution 0
                    with open("models/tuned_model_"+engine+"_"+alg+"_"+date+".pkl", "rb") as fp:
                        tuned_model = pickle.load(fp)

                # Model evaluation
                y_pred = tuned_model.predict(X_test)
                y_pred = pd.DataFrame(y_pred)
                # Simple regularization: replace by 9 if the predicted value is greater than 9 (Saaty's scale max value)
                y_pred[y_pred > 9] = 9
                if mcdm_pairwise_method == 'AHP':
                    y_pred = y_pred.astype(int)

                results[alg][engine]['accuracy'].append(accuracy(y_pred, y_test))
                results[alg][engine]['mse'].append(mse(y_pred, y_test))
                results[alg][engine]['rmse'].append(rmse(y_pred, y_test))
                results[alg][engine]['r2'].append(r2(y_pred, y_test))
                results[alg][engine]['mape'].append(mape(y_pred, y_test))

                # Merge the predictions of the df train and df test
                df_merged = merge_matrices(N_Q, pc_matrix, y_pred)

                # Employ MCDM method in the predicted (mixed with preferences) dataset
                if mcdm_pairwise_method == 'AHP':
                    rank_predicted = Mcdm_ranking().ahp_ranking(pc_matrix=df_merged, weights=None, nrow=npop, nobj=nobj)
                elif mcdm_pairwise_method == 'Promethee':
                    rank_predicted = Mcdm_ranking().promethee_ii_ranking(pc_matrix=df_merged, weights=None, nobj=nobj, nrow=npop)
                else:
                    raise ValueError("MCDM not implemented")

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
            #end for that makes the recommendations
        #end number of executions
    #end for with the different recomm engines
#end for with the ML methods

# Save results
filename = open('results.pkl', 'wb')
pickle.dump(results, filename)
filename.close()

for alg in ['gbr', 'lasso', 'elasticnet', 'rf', 'ridge']:
    # Euclidean recommendation
    euc = pd.DataFrame(results[alg]['personalized']['tau'])
    euc['Recommendation'] = tipo_recomendacao
    euc['iteracao'] = list(range(0, int(total_samples_per_rec / n_rec))) * n_executions

    euc_ = pd.DataFrame(results[alg]['personalized']['rho'])
    euc_['Recommendation'] = tipo_recomendacao
    euc_['iteracao'] = list(range(0, int(total_samples_per_rec / n_rec))) * n_executions

    euc_rmse = pd.DataFrame(results[alg]['personalized']['rmse'])
    euc_rmse['Recommendation'] = tipo_recomendacao
    euc_rmse['iteracao'] = list(range(0, int(total_samples_per_rec / n_rec))) * n_executions

    # Aleatory recommendation
    ale = pd.DataFrame(results[alg]['aleatory']['tau'])
    ale['Recommendation'] = 'Aleatory'
    ale['iteracao'] = list(range(0, int(total_samples_per_rec / n_rec))) * n_executions

    ale_ = pd.DataFrame(results[alg]['aleatory']['tau'])
    ale_['Recommendation'] = 'Aleatory'
    ale_['iteracao'] = list(range(0, int(total_samples_per_rec / n_rec))) * n_executions

    ale_rmse = pd.DataFrame(results[alg]['aleatory']['rmse'])
    ale_rmse['Recommendation'] = 'Aleatory'
    ale_rmse['iteracao'] = list(range(0, int(total_samples_per_rec / n_rec))) * n_executions

    tau = pd.concat([euc, ale])
    rho = pd.concat([euc_, ale_])
    rmse = pd.concat([euc_rmse, ale_rmse])

    fig, axes = plt.subplots(3, 1, sharex=True)
    fig.suptitle(mcdm_pairwise_method + '--' + alg)
    for i, k in enumerate([tau, rho, rmse]):
        sns.boxplot(data=k, hue='Recommendation', x='iteracao', y=0, ax=axes[i])
    axes[0].legend(loc="upper right", prop={'size': 8})
    axes[1].get_legend().remove()
    axes[2].get_legend().remove()

    axes[0].set_ylabel('Tau')
    axes[0].set_xlabel('')
    axes[1].set_ylabel('Rho')
    axes[1].set_xlabel('')
    axes[2].set_ylabel('RMSE')
    axes[2].set_xlabel('Iteration')

    plt.show()

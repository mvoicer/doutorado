import pickle
import pandas as pd
import seaborn as sns
from pre_processing import is_testing
from matplotlib import pyplot as plt

approach_to_analyze = 'Euclidean'
mcdm = 'Promethee'

file = open("results/results_"+approach_to_analyze+"_"+mcdm+"_13122021.pkl",'rb')
# file=open("results/results_lasso_personalized_Cosine_AHP_12122021.pkl", 'rb')
results = pickle.load(file)
file.close()

n_executions, total_samples_per_rec, n_rec, CV, test_size, top_n, max_sample = is_testing(True)

#TODO: Optimize code.
for alg in ['gbr', 'rf']:
    # Euclidean recommendation
    euc = pd.DataFrame(results[alg]['personalized']['tau'])
    euc['Recommendation'] = approach_to_analyze
    euc['iteracao'] = list(range(0, int(total_samples_per_rec / n_rec))) * n_executions

    euc_ = pd.DataFrame(results[alg]['personalized']['rho'])
    euc_['Recommendation'] = 'Euclidean'
    euc_['iteracao'] = list(range(0, int(total_samples_per_rec / n_rec))) * n_executions

    euc_rmse = pd.DataFrame(results[alg]['personalized']['rmse'])
    euc_rmse['Recommendation'] = 'Euclidean'
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

    # tau = pd.concat([euc, ale]).groupby(by=['Recommendation', 'iteracao'])[0].median().reset_index()
    # rho = pd.concat([euc_, ale_]).groupby(by=['Recommendation', 'iteracao'])[0].median().reset_index()
    # rmse = pd.concat([euc_rmse, ale_rmse]).groupby(by=['Recommendation', 'iteracao'])[0].median().reset_index()

    tau = pd.concat([euc, ale])
    rho = pd.concat([euc_, ale_])
    rmse = pd.concat([euc_rmse, ale_rmse])

    fig, axes = plt.subplots(3, 1, sharex=True)
    fig.suptitle(mcdm + '--' + alg)
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


import joblib
import joblib as joblib
import pandas as pd
import numpy as np
import seaborn as sns
from params import *
from matplotlib import pyplot as plt

n_executions, total_samples_per_rec, max_sample, CV = load_params()

# list_metrics = ['tau', 'rho', 'mse', 'rmse', 'r2', 'mape']
mcdm = ['ahp']
percent_random = '0.5'
similarity_measure = 'Euclidean'
ml_method = 'rf'
n_rec = 5
eixo_y = ['tau']
date = 'opa1'

data = pd.DataFrame()
for ITEM in mcdm:
    results = joblib.load('results/' + (str(ITEM) + '_' + str(percent_random) + 'initrandom_sim' +
                                        similarity_measure + '_' + ml_method + '_' + date).lower() + '.gz')
    data['Iteration'] = list(range(0, int(total_samples_per_rec / n_rec))) * n_executions
    data.loc[:, ITEM] = results[eixo_y[0]]
    df = pd.concat([data])

g = sns.lineplot(data=pd.melt(df, ['Iteration']),
                 x='Iteration',
                 y='value',
                 hue='variable')
# g.legend(loc="upper right", prop={'size': 10})
g.legend('')
# g.set_title(to_evaluate[0].capitalize())
g.set_ylabel(eixo_y[0].capitalize())
g.set_xlabel('Iteration')
# g.set_ylim([0, 2])
g.set_xticks(np.arange(df.shape[0] / n_executions))
plt.show()

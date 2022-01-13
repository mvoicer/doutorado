

list_metrics = ['tau', 'rho', 'accuracy', 'mse', 'rmse', 'r2', 'mape']

# Apenas checa se está fazendo testes no algoritmo principal ou não (neste caso, usa o dataframe completo)
def load_params():
    n_executions = 3                # número de vezes que vai rodar o algoritmo para tirar a média
    total_samples_per_rec = 51      # total de amostras a serem avaliadas a cada n_rec
    max_sample = 120
    CV=5                            # number of cross-validation
    return n_executions, total_samples_per_rec, max_sample, CV

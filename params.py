

list_metrics = ['tau', 'rho', 'mse', 'rmse', 'r2', 'mape', 'mcdm']

# Apenas checa se está fazendo testes no algoritmo principal ou não (neste caso, usa o dataframe completo)
# def load_params():
#     n_executions = 5
#     total_samples_per_rec = 150
#     CV=3                            # number of cross-validation
#     return n_executions, total_samples_per_rec, max_sample, CV

def load_params():
    n_executions = 5                # número de vezes que vai rodar o algoritmo para tirar a média
    total_samples_per_rec = 76      # total de amostras a serem avaliadas a cada n_rec
    CV=4                            # number of cross-validation
    return n_executions, total_samples_per_rec, CV

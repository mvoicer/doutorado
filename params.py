

list_metrics = ['tau', 'rho', 'mse', 'rmse', 'r2', 'mape', 'mcdm']

# Apenas checa se está fazendo testes no algoritmo principal ou não (neste caso, usa o dataframe completo)
# def load_params():
#     n_executions = 5
#     total_samples_per_rec = 150
#     max_sample = 200
#     CV=3                            # number of cross-validation
#     return n_executions, total_samples_per_rec, max_sample, CV

def load_params():
    n_executions = 4                # número de vezes que vai rodar o algoritmo para tirar a média
    total_samples_per_rec = 200      # total de amostras a serem avaliadas a cada n_rec
    max_sample = 200
    CV=3                            # number of cross-validation
    return n_executions, total_samples_per_rec, max_sample, CV

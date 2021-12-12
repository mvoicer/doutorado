if True:
    n_executions = 4                # número de vezes que vai rodar o algoritmo para tirar a média
    total_samples_per_rec = 51      # total de amostras a serem avaliadas a cada n_rec
    n_rec: int = 5                  # numero de amostras que são apresentadas ao decisor por vez
    CV: int = 5                     # number of cross-validation
    test_size: float = 0.2          # 80% train and 20% test
    top_n = n_rec                   # top n solutions
    max_sample = 120
if not True:
    n_executions = 2                # número de vezes que vai rodar o algoritmo para tirar a média
    total_samples_per_rec = 31      # total de amostras a serem avaliadas a cada n_rec
    n_rec: int = 5                  # numero de amostras que são apresentadas ao decisor por vez
    CV: int = 2                     # number of cross-validation
    test_size: float = 0.2          # 80% train and 20% test
    top_n = n_rec                   # top n solutions
    max_sample = 51

list_algs = ['gbr', 'lasso', 'elasticnet', 'rf', 'ridge']
recomm_engines = ['aleatory', 'personalized']
list_metrics = ['tau', 'rho', 'accuracy', 'mse', 'rmse', 'r2', 'mape']

def initialize_results():
    results = {}
    for alg in list_algs:
        results[alg] = {}
        for eng in recomm_engines:
            results[alg][eng] = {}
            for metr in list_metrics:
                results[alg][eng][metr] = []
    return results
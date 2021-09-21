n_executions = 2                # número de vezes que vai rodar o algoritmo para tirar a média
TOTAL = 31                      # total de amostras a serem avaliadas a cada N_REC
N_REC: int = 5                  # n_samples to be evaluated
CV: int = 5                     # number of cross-validation
test_size: float = 0.2          # 80% train and 20% test
TOP_N = N_REC                   # top n solutions
tau_all: float = 1.

results = {'aleatory': {'tau': [],
                        'cosine': [],
                        'accuracy': [],
                        'mse': [],
                        'rmse': [],
                        'r2': [],
                        'mape': []
                        },
           'euclidean': {'tau': [],
                         'cosine': [],
                         'accuracy': [],
                         'mse': [],
                         'rmse': [],
                         'r2': [],
                         'mape': []
                         }
           }
recomm_engines = ['aleatory', 'euclidean']
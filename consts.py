if not True:
    n_executions = 3                # número de vezes que vai rodar o algoritmo para tirar a média
    total_samples_per_rec = 51      # total de amostras a serem avaliadas a cada n_rec
    n_rec: int = 5                  # numero de amostras que são apresentadas ao decisor por vez
    CV: int = 5                     # number of cross-validation
    test_size: float = 0.2          # 80% train and 20% test
    top_n = n_rec                   # top n solutions
    max_sample = 100
if True:
    n_executions = 3                # número de vezes que vai rodar o algoritmo para tirar a média
    total_samples_per_rec = 11      # total de amostras a serem avaliadas a cada n_rec
    n_rec: int = 2                  # numero de amostras que são apresentadas ao decisor por vez
    CV: int = 2                     # number of cross-validation
    test_size: float = 0.2          # 80% train and 20% test
    top_n = n_rec                   # top n solutions
    max_sample = 15

results = {'aleatory': {'tau': [],
                        'rho': [],
                        'accuracy': [],
                        'mse': [],
                        'rmse': [],
                        'r2': [],
                        'mape': []
                        },
           'personalized': {'tau': [],
                         'rho': [],
                         'accuracy': [],
                         'mse': [],
                         'rmse': [],
                         'r2': [],
                         'mape': []
                         }
           }
recomm_engines = ['aleatory', 'personalized']
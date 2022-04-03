from run_recommender import run_recommender

# NSGAIII_GPD03_M2 -> 2 objetivos, convexa e multimodal
# NSGAIII_GPD04_M3 -> 3 objetivos
# NSGAIII_GPD04_M2 -> 2 objetivos, concava

results = run_recommender(dataframe='NSGAIII_GPD04_M3',
                          n_rec=5,                                      # int: qtd solutions to be evaluated per time
                          mcdm_method='AHP',                             # AHP, Promethee
                          weights=None,
                          cost_benefit=None,
                          percent_random=0.5,                           # {0-1}% float. Aleatory solutions to be recomm
                          initial_recomm='cluster',                     # aleatory, cluster
                          similarity_measure='Euclidean',               # Cosine, Euclidean
                          ml_method='rf',                               #'gbr', 'lasso', 'elasticnet', 'rf', 'ridge'
                          date='opa11',
                          plot_pareto_front=False,
                          plot_recommended_solutions=False)

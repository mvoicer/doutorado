from run_recommender import run_recommender


results = run_recommender(dataframe='NSGAIII_GPD04_M3',
                          n_rec=5,                                      # int: qtd solutions to be evaluated per time
                          mcdm_method='AHP',                             # AHP, Promethee
                          weights=None,
                          cost_benefit=None,
                          percent_random=0.5,                           # {0-1}% float. Aleatory solutions to be recomm
                          initial_recomm='cluster',                     # rand, cluster
                          similarity_measure='euc',                     # cos, euc
                          ml_method='rf',                               #'gbr', 'lasso', 'elast', 'rf', 'ridge'
                          date='10102022',
                          n_executions = n_executions,
                          total_samples_per_rec = total_samples_per_red,
                          CV = CV)

from run_recommender import run_recommender


results = run_recommender(dataframe='NSGAIII_GPD04_M3',
                          n_rec=5,                                      # int: qtd solutions to be evaluated per time
                          mcdm_method='AHP',                            # AHP
                          weights=None,
                          cost_benefit=None,
                          percent_random=0.5,                           # {0-1}% float.
                          initial_recomm='cluster',                     # rand, kmeans, agglomerative
                          similarity_measure='euclidean',               # cosine, euclidean, manhattan
                          ml_method='rf',                               # gbr, rf
                          date='10102022',
                          n_executions = n_executions,
                          total_samples_per_rec = total_samples_per_red,
                          CV = CV)

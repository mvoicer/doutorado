from recommendation import run_recommender

# NSGAIII_GPD03_M2 -> 2 objetivos, convexa e multimodal
# NSGAIII_GPD04_M3 -> 3 objetivos
# NSGAIII_GPD04_M2 -> 2 objetivos, concava

results = run_recommender(dataframe='NSGAIII_GPD04_M2',
                          recomm_approach='Euclidean',              # Cosine, Euclidean
                          mcdm_method='Promethee',                  # AHP, Promethee
                          initial_recomm='cluster',                 # aleatory, cluster
                          weights=None,
                          recomm_style='most_similar',              # most_similar, half_similar_half_dissimilar
                          testing=False,
                          date='21122021',
                          print_pf=False)
print(results)

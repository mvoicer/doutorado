from sklearn.metrics import pairwise_distances

def calculate_similarities(df_obj, simm_approach):
    """
    Summary:
        Calculate the distance among all the solutions based on 'simm_approach'.
    Parameters:
        df_obj: dataframe with the values in objective space
        simm_approach: similarity measure to be used.
            - 'cos': cosine similarity
            - 'euc': euclidean distance
            - 'man': manhattan distance
    Returns:
        dataframe with the distances
    """
    if simm_approach == 'cos':
        #TODO: Ver distancia e similaridade.
        #https://stackoverflow.com/questions/58381092/difference-between-cosine-similarity-and-cosine-distance
        #https://stackoverflow.com/questions/18424228/cosine-similarity-between-2-number-lists
        df_dist = 1 - pairwise_distances(df_obj, metric='cosine')
    elif simm_approach == 'euc':
        df_dist = pairwise_distances(df_obj, metric='euclidean')
    elif simm_approach == 'manhattan':
        df_dist = pairwise_distances(df_obj, metric='manhattan')
    else:
        raise ValueError('Distance indicated is not implemented')
    return df_dist

from sklearn.metrics import pairwise_distances
import pandas as pd


def calculate_similarities(df_obj, zeta):
    """
    Summary:
        Calculate the distance among all the solutions.
    Parameters:
        df_obj: dataframe with the values in objective space
        zeta: similarity distance between vectors u and v.
            - 'cosine': cosine similarity
            - 'euclidean': euclidean distance
            - 'manhattan': manhattan distance
            - 'chebyshev': Chebyshev distance
    Returns:
        dataframe with the distances
    """
    if zeta not in ['euclidean', 'cosine', 'chebyshev', 'manhattan', 'minkowski_2', 'minkowski_05']:
        raise ValueError(f'Invalid simmilarity distance: {zeta}')
    elif zeta == 'euclidean':
        df_dist = pairwise_distances(df_obj.values, metric='euclidean')
    elif zeta == 'cosine':
        df_dist = pairwise_distances(df_obj.values, metric='cosine')
    elif zeta == 'chebyshev':
        df_dist = pairwise_distances(df_obj.values, metric='chebyshev')
    elif zeta == 'manhattan':
        df_dist = pairwise_distances(df_obj.values, metric='manhattan')
    elif zeta == 'minkowski_2':
        df_dist = pairwise_distances(df_obj.values, metric='minkowski', p=2)
    else:
        df_dist = pairwise_distances(df_obj.values, metric='minkowski', p=.5)
    return pd.DataFrame(df_dist)

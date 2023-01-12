import numpy as np
from random import randrange
from sklearn.cluster import KMeans, AgglomerativeClustering


def clusterization(df, cluster_technique, tam, visualization=False, **kwargs):
    """
    A function for clustering data using KMeans or Agglomerative Clustering.

    Parameters:
    - df: DataFrame, Data to be clustered.
    - cluster_technique: str, 'kmeans' or 'agglomerative'. The clustering technique to be used.
    - tam: int, Number of clusters.
    - visualization: bool, If set to True, the function will plot the clusters using matplotlib.
    - kwargs: Additional arguments to pass to the clustering algorithm, such as initialization method or number of iterations.

    Returns:
    - A list of randomly chosen samples from each cluster.
    """

    if cluster_technique == 'kmeans':
        model = KMeans(n_clusters=tam, **kwargs)
    elif cluster_technique == 'agglomerative':
        model = AgglomerativeClustering(n_clusters=tam, **kwargs)
    else:
        raise ValueError("Cluster technique {} not implemented".format(cluster_technique))
        
    labels = model.fit_predict(df)
    unique_labels = np.unique(labels)

    if visualization:
        import matplotlib.pyplot as plt
        for label in unique_labels:
            plt.scatter(df[labels==label, 0], df[labels==label, 1])
        plt.show()

    chosen = []
    for unq in unique_labels:
        rand = randrange(len(np.where(labels == unique_labels[unq])[0]))
        chosen.append(np.where(labels == unique_labels[unq])[0][rand])
    return chosen

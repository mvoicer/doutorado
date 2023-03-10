import numpy as np
from random import randrange
from sklearn.cluster import KMeans, AgglomerativeClustering


def clusterization(df, cluster_technique, tam):
    """
    A function for clustering data using KMeans or Agglomerative technique.

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
        model = KMeans(n_clusters=tam)
    elif cluster_technique == 'agglomerative':
        model = AgglomerativeClustering(n_clusters=tam)
    else:
        raise ValueError("Cluster technique {} not implemented".format(cluster_technique))

    labels = model.fit_predict(df)
    unique_labels = np.unique(labels)

    chosen = []
    for unq in unique_labels:
        rand = randrange(len(np.where(labels == unique_labels[unq])[0]))
        chosen.append(np.where(labels == unique_labels[unq])[0][rand])
    return chosen

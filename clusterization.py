import numpy as np
from random import randrange
from sklearn.cluster import KMeans


def clusterization(df, type, tam):
    if type == 'kmeans':
        kmeans = KMeans(n_clusters=tam,
                        init='k-means++').fit(df)
        labels = kmeans.labels_
        unique_labels = np.unique(labels)

        chosen = []
        for unq in unique_labels:
            rand = randrange(len(np.where(labels == unique_labels[unq])[0]))
            chosen.append(np.where(labels == unique_labels[unq])[0][rand])
        return chosen
    else:
        raise ValueError("Cluster technique not implemented")
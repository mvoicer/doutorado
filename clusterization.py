import numpy as np
from random import randrange
from sklearn.cluster import KMeans


def clusterization(df, type, n_solutions):
    if type == 'kmeans':
        kmeans = KMeans(n_clusters=n_solutions,
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














# def generateVectors(dim, n_vetores):
#     def ncr(n, r):
#         r = min(r, n - r)
#         numer = reduce(op.mul, range(n, n - r, -1), 1)
#         denom = reduce(op.mul, range(1, r + 1), 1)
#         return numer / denom
#
#     nRef = int(ncr((dim + n_vetores - 1), (dim - 1)))
#
#     vectors = np.zeros((dim, nRef))
#     combinations = np.array([(i) for i in combinations_with_replacement(range(1, (dim + 1)), n_vetores)])
#
#     for i in range(dim):
#         aux = np.array(list(map(lambda x: x == (i + 1), (combinations))))
#         for j in range(nRef):
#             vectors[i, j] = sum(aux[j,])
#
#     print(np.transpose((vectors / n_vetores)) * 2)
#
#
# generateVectors(2, 3)

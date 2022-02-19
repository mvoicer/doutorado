import random
from clusterization import clusterization


def make_recommendation(most_similar, Q, n_rec, qtd_to_add):
    # Recommend the closest solutions
    q_to_add = []
    [q_to_add.append(x) for x in most_similar if (len(q_to_add) <= qtd_to_add) & (x not in Q)]

    # Recommend aleatory solutions
    randomized = list(set(most_similar) - set(Q))
    random.shuffle(randomized)
    [q_to_add.append(y) for y in randomized if len(q_to_add) <= (n_rec-1)]

    return q_to_add


def initial_recommendation(type_rec, indexes, df_obj, length):
    if type_rec == 'aleatory':
        temp = indexes.copy()
        random.shuffle(temp)
        Q = temp[0:length]
    elif type_rec == 'cluster':
        Q = clusterization(df=df_obj, type='kmeans', tam=length)
    N_Q = [x for x in indexes if x not in Q]
    return Q, N_Q
import random
from clusterization import clusterization


def make_recommendation(most_similar, Q, tam, qtd_to_add):
    # Recommend the closest solutions
    for q in most_similar:
        if (len(Q) <= (tam - qtd_to_add)) & (q not in Q):
            Q.append(q)
    # Recommend aleatory solutions
    randomized = most_similar.copy()
    random.shuffle(randomized)
    for _ in most_similar:
        # randomized = most_similar.copy()
        # random.shuffle(randomized)
        for p in randomized:
            if (len(Q) <= tam) & (p not in Q):
                Q.append(p)
            else:
                continue
    return Q


def initial_recommendation(type, indexes, df_obj, length):
    if type == 'aleatory':
        temp = indexes.copy()
        random.shuffle(temp)
        Q = temp[0:length]
    elif type == 'cluster':
        Q = clusterization(df=df_obj, type='kmeans', tam=length)
    N_Q = [x for x in indexes if x not in Q]
    return Q, N_Q
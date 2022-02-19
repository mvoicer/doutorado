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

    for p in randomized:
        if (len(Q) <= (tam-1)) & (p not in Q):
            Q.append(p)
        else:
            continue
    return Q


def initial_recommendation(type_rec, indexes, df_obj, length):
    if type_rec == 'aleatory':
        temp = indexes.copy()
        random.shuffle(temp)
        Q = temp[0:length]
    elif type_rec == 'cluster':
        Q = clusterization(df=df_obj, type='kmeans', tam=length)
    N_Q = [x for x in indexes if x not in Q]
    return Q, N_Q
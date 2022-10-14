import random
from clusterization import clusterization


def make_recommendation(most_similar, Q, n_rec, qtd_aleatory_to_add):
    """
    Summary:
        Recommend the closest solutions
    Parameters:
        most_similar:           closest/similar solutions ranked highest in the previous iteration
        Q:                      list of the current solutions
        n_rec:                  number of solutions to be recommended per iteration
        qtd_aleatory_to_add:    quantity of solutions to be recommended in this iteration
    Returns:
        List of solutions to be recommended in the current iteration
    """
    new_recommendation = []
    [new_recommendation.append(x) for x in most_similar if (len(new_recommendation) < (n_rec - qtd_aleatory_to_add)) & (x not in Q)]

    # Recommend aleatory solutions
    recomm_aleatory = list(set(most_similar) - set(Q))
    random.shuffle(recomm_aleatory)
    [new_recommendation.append(y) for y in recomm_aleatory if len(new_recommendation) <= (n_rec-1)]

    return new_recommendation[:-1]


def initial_recommendation(type_rec, indexes, df_obj, length):
    """
    Summary:
        Define the solutions to be chosen in the 1st iteration
    Parameters:
        type_rec:
            rand:       aleatory recommendations
            cluster:    use kmeans technique and select 1 solution per cluster
        indexes:    indexes of the variables in decision space
        df_obj:     solutions in solutions space
        length:     n_rec- number of solutions to be recommended per iteration
        n_rec:                  number of solutions to be recommended per iteration
    """
    if type_rec == 'rand':
        temp = indexes.copy()
        random.shuffle(temp)
        Q = temp[0:length]
    elif type_rec == 'cluster':
        Q = clusterization(df=df_obj, cluster_technique='kmeans', tam=length)
    N_Q = [x for x in indexes if x not in Q]
    return Q, N_Q

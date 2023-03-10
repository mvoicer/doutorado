from functools import reduce
import math


def aggregate_weights(weights):
    """ Aggregate the weights of different decision-makers
    Parameters:
        weights :   list of lists
            Each internal list represents the weights/preferences of each decision-maker
    """
    return [sum(i) for i in zip(*weights)] / len(weights)


def aggregate_weights_geometric_mean(weights):
    # Aggregate the weights of different decision-makers using geometric mean
    produto = reduce(lambda x, y: [x[i]*y[i] for i in range(len(x))], weights)
    return math.pow(reduce(lambda x, y: x*y, produto), 1/len(weights))

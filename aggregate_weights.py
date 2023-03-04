


def aggregate_weights(weights):
    """ Aggregate the weights of different decision-makers
    Parameters:
        weights :   list of lists
            Each internal list represents the weights/preferences of each decision-maker
    """
    return [sum(i) for i in zip(*weights)] / len(weights)

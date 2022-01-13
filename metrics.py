import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score


# https://stackoverflow.com/questions/47648133/mape-calculation-in-python
def mape(y_pred, y_true):
    real, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((real - y_pred) / real))*100

# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html
def accuracy(y_pred, y_true):
    return accuracy_score(y_true, y_pred)

# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html
def mse(y_pred, y_true):
    return mean_squared_error(y_true, y_pred, squared=True)

# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html
def rmse(y_pred, y_true):
    return mean_squared_error(y_true, y_pred, squared=False)

# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html
def r2(y_pred, y_true):
    return r2_score(y_true, y_pred)
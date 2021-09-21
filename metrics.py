import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score


def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true))


def accuracy(pred, real):
    return np.mean(np.mean(1 - abs(pd.DataFrame(pred.values) - pd.DataFrame(real.values) / pd.DataFrame(pred.values))))


def mse(pred, real):
    return mean_squared_error(pd.DataFrame(real.values), pd.DataFrame(pred.values), squared=True)


def rmse(pred, real):
    return mean_squared_error(pd.DataFrame(real.values), pd.DataFrame(pred.values), squared=False)


def r2(pred, real):
    return r2_score(pd.DataFrame(real.values), pd.DataFrame(pred.values))


def mape(pred, real):
    return mean_absolute_percentage_error(pd.DataFrame(real.values), pd.DataFrame(pred.values))

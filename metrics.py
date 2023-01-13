from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error


# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_absolute_percentage_error.html
def mape(y_pred, y_true):
    return mean_absolute_percentage_error(y_true, y_pred,
                                          multioutput='uniform_average')


# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html
def mse(y_pred, y_true):
    return mean_squared_error(y_true, y_pred,
                              multioutput='uniform_average',
                              squared=True)

# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html
def rmse(y_pred, y_true):
    return mean_squared_error(y_true, y_pred,
                              multioutput='uniform_average',
                              squared=False)

# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html
def r2(y_pred, y_true):
    return r2_score(y_true, y_pred,
                    multioutput='uniform_average')

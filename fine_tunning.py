from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import GradientBoostingRegressor


def fine_tunning(CV, X, y):
    model = MultiOutputRegressor(GradientBoostingRegressor(loss='ls',
                                                           learning_rate=0.001,
                                                           n_estimators=500,
                                                           subsample=1.0,
                                                           criterion='friedman_mse',
                                                           min_samples_split=10,
                                                           min_samples_leaf=1,
                                                           min_weight_fraction_leaf=0.0,
                                                           max_depth=30,
                                                           min_impurity_decrease=0.0,
                                                           min_impurity_split=None,
                                                           init=None,
                                                           random_state=42,
                                                           max_features=None,
                                                           alpha=0.9,
                                                           verbose=0,
                                                           max_leaf_nodes=None,
                                                           warm_start=False,
                                                           validation_fraction=0.1,
                                                           n_iter_no_change=None,
                                                           tol=0.0001,
                                                           ccp_alpha=0.0))

    hyperparameters = dict(estimator__learning_rate=[0.05],
                           estimator__loss=['ls'],
                           estimator__n_estimators=[500, 1000],
                           estimator__criterion=['friedman_mse'],
                           estimator__min_samples_split=[7, 10],
                           estimator__max_depth=[15, 30],
                           estimator__min_samples_leaf=[1, 2],
                           estimator__min_impurity_decrease=[0],
                           estimator__max_leaf_nodes=[5, 30])

    randomized_search = RandomizedSearchCV(model,
                                           hyperparameters,
                                           random_state=42,
                                           n_iter=5,
                                           scoring=None,
                                           n_jobs=-1,
                                           refit=True,
                                           cv=CV,
                                           verbose=True,
                                           pre_dispatch='2*n_jobs',
                                           error_score='raise',
                                           return_train_score=True)

    hyperparameters_tuning = randomized_search.fit(X, y)
    print('Best Parameters = {}'.format(hyperparameters_tuning.best_params_))

    tuned_model = hyperparameters_tuning.best_estimator_

    return tuned_model

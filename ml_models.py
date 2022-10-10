from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import MultiTaskLassoCV, LassoCV, Ridge
from sklearn import linear_model

def fine_tunning(CV, X, y, algorithm):
    if algorithm == 'gbr':
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
                               estimator__n_estimators=[50, 500, 1000, 2000],
                               estimator__criterion=['friedman_mse'],
                               estimator__min_samples_split=[7, 10],
                               estimator__max_depth=[15, 30],
                               estimator__min_samples_leaf=[1, 2, 5, 10],
                               estimator__min_impurity_decrease=[0],
                               estimator__max_leaf_nodes=[5, 30])
        randomized_search = RandomizedSearchCV(model,
                                               hyperparameters,
                                               random_state=42,
                                               n_iter=60,
                                               scoring=None,
                                               n_jobs=-1,
                                               refit=True,
                                               cv=CV,
                                               verbose=True,
                                               pre_dispatch='2*n_jobs',
                                               error_score='raise',
                                               return_train_score=True)
        hyperparameters_tuning = randomized_search.fit(X, y)
        tuned_model = hyperparameters_tuning.best_estimator_
        return tuned_model


    elif algorithm == 'lasso':
        model = MultiOutputRegressor(LassoCV(n_jobs=-1,
                                                      cv=CV,
                                                      verbose=True,
                                                      eps=0.00001,
                                                      n_alphas=100,
                                                      max_iter=6000000))
        hyperparameters = dict(estimator__eps=[.0001, .00001, .01, .0000001],
                               estimator__n_alphas=[10, 50, 100, 200],
                               estimator__normalize=[False],
                               estimator__max_iter=[50000, 5000000, 6000000],
                               estimator__selection=['cyclic','random'])
        randomized_search = RandomizedSearchCV(model,
                                               hyperparameters,
                                               random_state=42,
                                               n_iter=60,
                                               scoring=None,
                                               n_jobs=-1,
                                               refit=True,
                                               cv=CV,
                                               verbose=True,
                                               pre_dispatch='2*n_jobs',
                                               error_score='raise',
                                               return_train_score=True)
        hyperparameters_tuning = randomized_search.fit(X, y)
        tuned_model = hyperparameters_tuning.best_estimator_
        return tuned_model

    elif algorithm == 'elast':
        model = linear_model.MultiTaskElasticNet(alpha=.9,
                                                 max_iter=6000000,
                                                 tol=.00001,
                                                 warm_start=True,
                                                 random_state=42,
                                                 selection='cyclic')
        tuned_model = model.fit(X, y)
        return tuned_model

    elif algorithm == 'rf':
        model = MultiOutputRegressor(RandomForestRegressor(max_depth=2,
                                                           n_estimators=1000,
                                                           n_jobs=-1))
        hyperparameters = dict(estimator__n_estimators=[10, 50, 100, 1000],
                               estimator__max_features=["auto", "log2", "sqrt"],
                               estimator__bootstrap=[True, False],
                               estimator__max_depth=[5, 15, 20, 30, 80],
                               estimator__min_samples_leaf=[3, 4, 5],
                               estimator__min_samples_split=[2, 4, 8, 10, 12, 20],
                               estimator__warm_start=[True, False])
        randomized_search = RandomizedSearchCV(model,
                                               hyperparameters,
                                               random_state=42,
                                               n_iter=60,
                                               scoring=None,
                                               n_jobs=-1,
                                               refit=True,
                                               cv=CV,
                                               verbose=True,
                                               pre_dispatch='2*n_jobs',
                                               error_score='raise',
                                               return_train_score=True)
        hyperparameters_tuning = randomized_search.fit(X, y)
        tuned_model = hyperparameters_tuning.best_estimator_
        return tuned_model

    elif algorithm == 'ridge':
        model = MultiOutputRegressor(Ridge(random_state=42))
        hyperparameters = dict(estimator__alpha=[.1, .5, 1, 1.5, 2],
                               estimator__fit_intercept=[True, False],
                               estimator__normalize=[False],
                               estimator__max_iter=[5, 100, 200, 1000, 100000, 60000000],
                               estimator__tol=[.001, .00001, .000001],
                               estimator__solver=['auto','svd','cholesky','lsqr','sparse_cg','sag','saga'])
        randomized_search = RandomizedSearchCV(model,
                                               hyperparameters,
                                               random_state=42,
                                               n_iter=60,
                                               scoring=None,
                                               n_jobs=-1,
                                               refit=True,
                                               cv=CV,
                                               verbose=True,
                                               pre_dispatch='2*n_jobs',
                                               error_score='raise',
                                               return_train_score=True)
        hyperparameters_tuning = randomized_search.fit(X, y)
        tuned_model = hyperparameters_tuning.best_estimator_
        return tuned_model

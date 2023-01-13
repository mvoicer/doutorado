from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn import linear_model
import optuna


def fine_tunning(CV, X, y, algorithm):
    # if algorithm == 'gbr':
    #     model = MultiOutputRegressor(GradientBoostingRegressor())
    #     hyperparameters = dict(estimator__learning_rate=[0.05],
    #                            estimator__n_estimators=[50, 500, 1000],
    #                            estimator__min_samples_split=[7, 10],
    #                            estimator__max_depth=[15, 30],
    #                            estimator__min_samples_leaf=[1, 2, 10])
    #     grid_search = GridSearchCV(model,
    #                                hyperparameters,
    #                                cv=CV,
    #                                n_jobs=-1,
    #                                refit=True,
    #                                verbose=True,
    #                                return_train_score=True)
    #     hyperparameters_tuning = grid_search.fit(X, y)
    #     tuned_model = hyperparameters_tuning.best_estimator_
    #     return tuned_model

    if algorithm == 'gbr':
        # Cria o objeto de estudo
        study = optuna.create_study()

        # Define a função de objetivo
        def objective(trial):
            model = MultiOutputRegressor(GradientBoostingRegressor(
                learning_rate=trial.suggest_loguniform('learning_rate', 1e-4, 1e-1),
                n_estimators=trial.suggest_int('n_estimators', 50, 1000),
                min_samples_split=trial.suggest_int('min_samples_split', 2, 15),
                max_depth=trial.suggest_int('max_depth', 5, 30),
                min_samples_leaf=trial.suggest_int('min_samples_leaf', 1, 10)
            ))

            # Executa a validação cruzada
            scores = cross_val_score(model, X, y, cv=CV, n_jobs=-1)

            # Retorna a média das métricas de desempenho
            return np.mean(scores)

        # Executa a busca de hiperparâmetros
        study.optimize(objective, n_trials=100)

        # Recupera o melhor modelo
        best_model = study.best_trial.user_attrs['model']
        return best_model

    # elif algorithm == 'rf':
    #     model = MultiOutputRegressor(RandomForestRegressor())
    #     hyperparameters = dict(estimator__n_estimators=[10, 50, 100, 1000],
    #                            estimator__max_features=["log2", "sqrt"],
    #                            estimator__max_depth=[5, 15, 30, 80],
    #                            estimator__min_samples_leaf=[3, 4, 5],
    #                            estimator__min_samples_split=[2, 4, 10, 20])
    #     grid_search = GridSearchCV(model,
    #                                hyperparameters,
    #                                cv=CV,
    #                                n_jobs=-1,
    #                                refit=True,
    #                                verbose=True,
    #                                return_train_score=True)
    #     hyperparameters_tuning = grid_search.fit(X, y)
    #     tuned_model = hyperparameters_tuning.best_estimator_
    #     return tuned_model

    elif algorithm == 'rf':
        def objective(trial):
            n_estimators = trial.suggest_int('estimator__n_estimators', 10, 1000)
            max_features = trial.suggest_categorical('estimator__max_features', ["log2", "sqrt"])
            max_depth = trial.suggest_int('estimator__max_depth', 5, 80)
            min_samples_leaf = trial.suggest_int('estimator__min_samples_leaf', 3, 5)
            min_samples_split = trial.suggest_int('estimator__min_samples_split', 2, 20)
            criterion = trial.suggest_categorical('estimator__criterion',
                                                  ['squared_error', 'absolute_error', 'poisson'])

            model = MultiOutputRegressor(RandomForestRegressor(n_estimators=n_estimators,
                                                               max_features=max_features,
                                                               max_depth=max_depth,
                                                               min_samples_leaf=min_samples_leaf,
                                                               min_samples_split=min_samples_split,
                                                               criterion=criterion
                                                               ))

            scores = -cross_val_score(model, X, y, cv=CV, n_jobs=-1, scoring='neg_mean_absolute_error')
            return scores.mean()

        study = optuna.create_study()
        study.optimize(objective, n_trials=100)
        best_params = study.best_params
        model = MultiOutputRegressor(RandomForestRegressor(**best_params))
        model.fit(X, y)
        return model

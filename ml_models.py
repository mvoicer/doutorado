from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
import optuna
import numpy as np
from sklearn.model_selection import cross_val_score
from params import CV


def fine_tunning(X, y, algorithm):
    if algorithm == 'gbr':
        study = optuna.create_study()

        def objective(trial):
            model = MultiOutputRegressor(GradientBoostingRegressor(
                learning_rate=trial.suggest_loguniform('learning_rate', 1e-4, 1e-1),
                n_estimators=trial.suggest_int('n_estimators', 1, 1001, step=100),
                min_samples_split=trial.suggest_int('min_samples_split', 2, 7, step=1),
                max_depth=trial.suggest_int('max_depth', 2, 30, step=5),
                min_samples_leaf=trial.suggest_int('min_samples_leaf', 1, 5, step=1)
            ))

            scores = cross_val_score(model, X, y, cv=CV, n_jobs=-1)
            return np.mean(scores)

        # Executa a busca de hiperparâmetros
        study.optimize(objective, n_trials=100)

        # Recupera o melhor modelo
        best_model = study.best_trial.user_attrs.get('model')
        if best_model is not None:
            return best_model
        else:
            best_params = study.best_params
            best_model = MultiOutputRegressor(GradientBoostingRegressor(
                learning_rate=best_params['learning_rate'],
                n_estimators=best_params['n_estimators'],
                min_samples_split=best_params['min_samples_split'],
                max_depth=best_params['max_depth'],
                min_samples_leaf=best_params['min_samples_leaf']
            ))
            best_model.fit(X, y)
            study.best_trial.user_attrs['model'] = best_model
            return best_model

    elif algorithm == 'rf':
        def objective(trial):
            n_estimators = trial.suggest_int('n_estimators', 1, 1001, step=100)
            max_features = trial.suggest_categorical('max_features', ["log2", "sqrt"])
            max_depth = trial.suggest_int('max_depth', 3, 30, step=5)
            min_samples_leaf = trial.suggest_int('min_samples_leaf', 3, 5, step=1)
            min_samples_split = trial.suggest_int('min_samples_split', 2, 5, step=1)
            criterion = trial.suggest_categorical('criterion',
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

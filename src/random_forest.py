import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectKBest
from sklearn.pipeline import Pipeline
import optuna

from functools import partial
from sklearn.feature_selection import mutual_info_regression

SEED = 42
H_MIN = 1e-8

mi_score = partial(mutual_info_regression, random_state=SEED)


def mascara(index, ini, fin):
    return (index >= ini) & (index <= fin)


def perdidas(y_true, h):
    # MSE y QLIKE diarios (QLIKE robusta de Patton, 2011)
    h = np.clip(np.asarray(h, dtype=float), H_MIN, None)
    y_true = np.asarray(y_true, dtype=float)
    return pd.DataFrame({"mse": (y_true - h)**2, "qlike": y_true / h + np.log(h)})


def crear_pipeline(k, score_func=None, **params_rf):
    # Pipeline: selección tipo filter (información mutua) + Random Forest
    return Pipeline([
        ("seleccion", SelectKBest(score_func=score_func or mi_score, k=k)),
        ("modelo", RandomForestRegressor(random_state=SEED, n_jobs=-1, **params_rf)),
    ])


def puntajes_fijos(scores):
    # score_func para SelectKBest que devuelve puntajes ya calculados
    return lambda X_, y_: np.asarray(scores)



def objetivo(trial, K_FINAL, MI_FOLD,CV, X, y):
    params = dict(
        n_estimators=trial.suggest_int("n_estimators", 100, 600, step=50),
        max_depth=trial.suggest_int("max_depth", 2, 20),
        min_samples_leaf=trial.suggest_int("min_samples_leaf", 5, 200, log=True),
        max_features=trial.suggest_float("max_features", 0.2, 1.0),
    )
    errores = []
    for i, (rep, fold, tr, va) in enumerate(CV):
        pipe = crear_pipeline(K_FINAL, score_func=puntajes_fijos(MI_FOLD[i]), **params)
        pipe.fit(X.iloc[tr], y.iloc[tr])                         # ajuste solo con el entrenamiento
        h_va = pipe.predict(X.iloc[va])                          # evaluación en la validación
        errores.append(perdidas(y.iloc[va], h_va)["qlike"].mean())

        trial.report(np.mean(errores), step=i)                   # poda si va peor que la mediana
        if trial.should_prune():
            raise optuna.TrialPruned()
    return float(np.mean(errores))
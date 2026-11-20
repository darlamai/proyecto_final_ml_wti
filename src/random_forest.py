# =============================================================================
# src/random_forest.py
# Funciones del modelo de machine learning: Random Forest con rezagos.
#
# Contenido:
#   - mi_score:        información mutua para regresión con semilla fija.
#   - mascara:         selecciona por fechas las filas de train o validación de un fold.
#   - perdidas:        MSE y QLIKE diarios entre la varianza pronosticada y Parkinson.
#   - crear_pipeline:  Pipeline de scikit-learn con selección tipo filter
#                      (SelectKBest + información mutua) y RandomForestRegressor.
#   - puntajes_fijos:  permite que SelectKBest use la información mutua ya calculada
#                      por fold, para no recalcularla en cada prueba de Optuna.
#   - objetivo:        función objetivo de Optuna: QLIKE promedio de validación en los
#                      10 folds de Time Series Cross-Validation, con poda por mediana.
#
# Uso de asistentes de IA (requisito de la rúbrica):
#   Partes de este código se generaron o adaptaron con ayuda de un asistente de IA: Claude  
# =============================================================================


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



# Prompt (Claude, 15-nov-2026): "Ahora dame el codigo para feature selection"
# Uso: información mutua con random_state fijo (el estimador usa vecinos más cercanos).

mi_score = partial(mutual_info_regression, random_state=SEED)

# Prompt (Claude, 15-nov-2026): "Ahora dame el codigo para feature selection"
def mascara(index, ini, fin):
    return (index >= ini) & (index <= fin)



# Prompt (Claude, 15-nov-2026): "una vez que tengo tsne, las variables seleccionadas, y todo,
#   ayudame con el codigo para el random forest con rezagos 2. Optimización con Optuna ...
#   minimizando la QLIKE de validación"
# Uso: QLIKE = y/h + ln(h), robusta al ruido del proxy de Parkinson.
# Referencias: [6]
def perdidas(y_true, h):
    # MSE y QLIKE diarios (QLIKE robusta de Patton, 2011)
    h = np.clip(np.asarray(h, dtype=float), H_MIN, None)
    y_true = np.asarray(y_true, dtype=float)
    return pd.DataFrame({"mse": (y_true - h)**2, "qlike": y_true / h + np.log(h)})


# Prompt (Claude, 15-nov-2026): "listo, ahora con esta inicializacion ayudame con el codigo para
#   continuar con el random forest ... debes usar pipelines, usar .fit en la parte de train y con
#   la de validacion emplear los hiperparametros"
# Uso: .fit ajusta la selección y el Random Forest solo con el train de cada fold;
#   .predict aplica ambos pasos a la validación.
# Referencias: [1], [2], [3]

def crear_pipeline(k, score_func=None, **params_rf):
    # Pipeline: selección tipo filter (información mutua) + Random Forest
    return Pipeline([
        ("seleccion", SelectKBest(score_func=score_func or mi_score, k=k)),
        ("modelo", RandomForestRegressor(random_state=SEED, n_jobs=-1, **params_rf)),
    ])

# Prompt (Claude, 15-nov-2026): "No se para que tengo la infromacion mutua precalcula por fold si
#   ya la calcule en mi feature selection"
# Uso: devuelve los puntajes ya calculados en el notebook de selección; el resultado es
#   idéntico a recalcular la información mutua en cada prueba de Optuna.
# Referencias: [2]

def puntajes_fijos(scores):
    # score_func para SelectKBest que devuelve puntajes ya calculados
    return lambda X_, y_: np.asarray(scores)

# Prompt (Claude, 15-nov-2026): "una vez que tengo tsne, las variables seleccionadas, y todo,
#   ayudame con el codigo para el random forest con rezagos 2. Optimización con Optuna ..."
# Prompt (Claude, 15-nov-2026): "listo, ahora con esta inicializacion ayudame con el codigo para
#   continuar con el random forest ..." (k fijo en 16, elegido en feature selection).
# Uso: QLIKE promedio de validación en los 10 folds; poda por mediana con trial.report.
#   Se llama con: study.optimize(lambda t: objetivo(t, K_FINAL, MI_FOLD, CV, X, y), ...)
# Referencias: [4], [5], [6]

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
# =============================================================================
# src/comparison.py
# Funciones para comparar estadísticamente los pronósticos de volatilidad del
# modelo econométrico (media constante – EGARCH(1,1,1)-t) y del Random Forest.
#
# Contenido:
#   - perdidas:          MSE y QLIKE diarios entre el pronóstico de varianza y el
#                        estimador de Parkinson (proxy de la varianza real).
#   - pruebas_pareadas:  prueba de Wilcoxon de rangos con signo y t pareada sobre
#                        las pérdidas de los 10 folds de Time Series Cross-Validation.
#   - diebold_mariano:   prueba de Diebold-Mariano sobre las pérdidas diarias, con
#                        varianza de largo plazo de Newey-West (HAC).
#
# Uso de asistentes de IA (requisito de la rúbrica):
#   Partes de este código se generaron o adaptaron con ayuda de un asistente de IA: Claude
# =============================================================================


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats



SEED = 42
H_MIN = 1e-6
ALFA = 0.05



# Prompt (Claude, 13-nov-2026): "osea con RF modelo la volatilidad y con Garch el precio del
#   retorno mas la volatilidad, entonces como seria la comparacion de modelos?"
# Prompt (Claude, 17-nov-2026): "ya termine de correr random forest como el ar-egarch, ahora
#   ayudame con el codigo para el cuaderno de comparacion de modelos, te adjunto los dos
#   cuadernos para que veas las salidas y lo que me sirve para cargar en mi notebook de
#   comparacion"
# Uso: pérdidas diarias. QLIKE = y/h + ln(h), robusta al ruido del proxy de Parkinson.
# Referencias: [1], [6]


def perdidas(y_true, h):
    # MSE y QLIKE diarios (la misma definición usada en los notebooks 03 y 06)
    h = np.clip(np.asarray(h, dtype=float), H_MIN, None)
    y_true = np.asarray(y_true, dtype=float)
    return pd.DataFrame({"mse": (y_true - h)**2, "qlike": y_true / h + np.log(h)})


def pruebas_pareadas(a, b, nombre):
    # Wilcoxon de rangos con signo y t pareada sobre las pérdidas por fold (a − b)
    w = stats.wilcoxon(a, b)
    t = stats.ttest_rel(a, b)
    d = np.asarray(a) - np.asarray(b)
    return {"métrica": nombre, "media EGARCH": np.mean(a), "media RF": np.mean(b),
            "diferencia media (EGARCH − RF)": d.mean(), "folds con RF mejor": int((d > 0).sum()),
            "Wilcoxon W": w.statistic, "Wilcoxon p": w.pvalue,
            "t pareada": t.statistic, "t p": t.pvalue}







# Prompt (Claude, 11-nov-2026): "osea dice x ejemplo WILCOXON, no que deba usar wilcoxon..."
#   (propuesta de Diebold-Mariano como prueba complementaria para series de tiempo).
# Prompt (Claude, 17-nov-2026): "ya termine de correr random forest como el ar-egarch, ahora
#   ayudame con el codigo para el cuaderno de comparacion de modelos..."
# Uso: H0 = E[d_t] = 0 (igual precisión de pronóstico), con d_t = L1 − L2.
#   Rezagos HAC por defecto: floor(4·(T/100)^(2/9)); pesos de Bartlett.
# Referencias: [4], [5], [6]


def diebold_mariano(L1, L2, max_lags=None):
    # Prueba de Diebold-Mariano. H0: ambos modelos tienen la misma precisión.
    # d_t = L1 − L2 (positivo → el modelo 2 tiene menor pérdida).
    # La varianza de la media de d_t se estima con Newey-West (HAC), porque las
    # diferencias de pérdida diarias están autocorrelacionadas.
    d = pd.Series(np.asarray(L1, dtype=float) - np.asarray(L2, dtype=float)).dropna().to_numpy()
    T = len(d)
    if max_lags is None:
        max_lags = int(np.floor(4 * (T / 100) ** (2 / 9)))
    u = d - d.mean()
    var_lr = u @ u / T
    for k in range(1, max_lags + 1):
        w = 1 - k / (max_lags + 1)                       # pesos de Bartlett
        var_lr += 2 * w * (u[k:] @ u[:-k]) / T
    estad = d.mean() / np.sqrt(var_lr / T)
    p = 2 * (1 - stats.t.cdf(abs(estad), df=T - 1))
    return {"T": T, "rezagos HAC": max_lags, "diferencia media": d.mean(), "DM": estad, "p-valor": p}
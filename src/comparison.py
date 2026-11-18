
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats



SEED = 42
H_MIN = 1e-6
ALFA = 0.05




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



import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from arch import arch_model

from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch


def obtener_residuos_estandarizados(res):
    """Devuelve z_t = residuo / desviación estándar condicional."""
    return pd.Series(res.std_resid).dropna()



def prueba_autocorrelacion_residuos(res, lags=(10, 20)):
    z = obtener_residuos_estandarizados(res)

    tabla = acorr_ljungbox(z, lags=list(lags), return_df=True)
    tabla.index.name = "Rezago"
    return tabla


def prueba_autocorrelacion_residuos_cuadrados(res, lags=(10, 20)):
    z2 = obtener_residuos_estandarizados(res) ** 2

    tabla = acorr_ljungbox(z2, lags=list(lags), return_df=True)
    tabla.index.name = "Rezago"
    return tabla


def graficar_residuos(res):
    residuos_std = pd.Series(res.std_resid).dropna()

    fig, ax = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

    residuos_std.plot(ax=ax[0], title="Residuos estandarizados")
    ax[0].axhline(0, color="black", linewidth=1)
    ax[0].set_ylabel(r"$z_t$")

    (residuos_std**2).plot(ax=ax[1], color="darkorange",
                           title="Residuos estandarizados al cuadrado")
    ax[1].set_ylabel(r"$z_t^2$")
    ax[1].set_xlabel("Fecha")

    plt.tight_layout()
    plt.show()


def graficar_distribucion_residuos(res):
    z = obtener_residuos_estandarizados(res)
    nu = res.params.get("nu")

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))

    # Histograma y distribución asumida
    ax[0].hist(z, bins=40, density=True, alpha=0.6, edgecolor="white")

    x = np.linspace(z.min(), z.max(), 300)

    if nu is not None and nu > 2:
        # t escalada para que tenga varianza 1
        escala = np.sqrt((nu - 2) / nu)
        distribucion = stats.t(df=nu, scale=escala)
        etiqueta = f"t estandarizada (ν={nu:.2f})"
    else:
        distribucion = stats.norm()
        etiqueta = "Normal estándar"

    ax[0].plot(x, distribucion.pdf(x), color="darkred", label=etiqueta)
    ax[0].set_title("Residuos estandarizados")
    ax[0].legend()

    # Gráfico Q–Q frente a la distribución asumida
    probabilidades = (np.arange(1, len(z) + 1) - 0.5) / len(z)
    cuantiles_teoricos = distribucion.ppf(probabilidades)
    cuantiles_observados = np.sort(z.to_numpy())

    ax[1].scatter(cuantiles_teoricos, cuantiles_observados, s=10, alpha=0.6)
    limites = [
        min(cuantiles_teoricos.min(), cuantiles_observados.min()),
        max(cuantiles_teoricos.max(), cuantiles_observados.max())
    ]
    ax[1].plot(limites, limites, "r--")
    ax[1].set_title("Gráfico Q–Q")
    ax[1].set_xlabel("Cuantiles teóricos")
    ax[1].set_ylabel("Cuantiles observados")

    plt.tight_layout()
    plt.show()


def prueba_ks_t_estandarizada(res):
    z = np.asarray(res.std_resid)
    z = z[np.isfinite(z)]

    nu = res.params["nu"]
    escala = np.sqrt((nu - 2) / nu)  # varianza igual a 1

    resultado = stats.kstest(
        z,
        stats.t(df=nu, loc=0, scale=escala).cdf
    )

    return {
        "estadístico_KS": resultado.statistic,
        "p-valor": resultado.pvalue
    }



def prueba_arch_remanente(res, lags=10):
    z = obtener_residuos_estandarizados(res)
    lm_stat, lm_p, f_stat, f_p = het_arch(z, nlags=lags)

    return pd.DataFrame({
        "estadístico": [lm_stat, f_stat],
        "p-valor": [lm_p, f_p]
    }, index=["ARCH-LM", "ARCH-LM (F)"])




def revisar_restricciones(res, tipo_vol="EGARCH"):
    """Revisa condiciones pertinentes para AR-EGARCH, no las de GARCH estándar.

    Para EGARCH(1,1,1), |beta[1]| < 1 es la condición usual para la
    estacionariedad de la recursión del logaritmo de la varianza. No implica
    por sí sola que la varianza incondicional de sigma² sea finita.
    """
    if tipo_vol.upper() not in {"EGARCH", "E-GARCH"}:
        raise ValueError("Esta función revisa EGARCH; use tipo_vol='EGARCH'.")

    p = res.params
    filas = []

    # En EGARCH, omega, alpha y gamma no tienen las restricciones de
    # no negatividad propias del GARCH estándar.
    beta_items = []
    for nombre, valor in p.items():
        coincidencia = re.fullmatch(r"beta\[(\d+)\]", nombre)
        if coincidencia:
            beta_items.append((int(coincidencia.group(1)), float(valor)))

    if beta_items:
        beta_items.sort()
        beta = np.array([valor for _, valor in beta_items])
        # Raíces de 1 - beta_1*z - ... - beta_q*z^q = 0.
        raices_beta = np.roots(np.r_[-beta[::-1], 1])
        radio_minimo = float(np.min(np.abs(raices_beta)))
        filas.append((
            "Estacionariedad de log(varianza): raíces fuera del círculo unitario",
            radio_minimo,
            bool(np.all(np.abs(raices_beta) > 1))
        ))
    else:
        filas.append(("Coeficientes beta de EGARCH disponibles", np.nan, False))

    if "nu" in p:
        filas.extend([
            ("ν > 2: varianza t finita", float(p["nu"]), bool(p["nu"] > 2)),
            ("ν > 4: cuarto momento t finito", float(p["nu"]), bool(p["nu"] > 4)),
        ])

    # Comprobar estacionariedad de la ecuación AR(p), si existe.
    coef_ar = []
    for nombre, valor in p.items():
        coincidencia = re.search(r"\[(\d+)\]$", nombre)
        if coincidencia and not nombre.startswith(("alpha", "beta", "gamma")):
            coef_ar.append((int(coincidencia.group(1)), float(valor)))

    if coef_ar:
        coef_ar.sort()
        phi = np.array([valor for _, valor in coef_ar])
        raices_ar = np.roots(np.r_[-phi[::-1], 1])
        filas.append((
            "Estacionariedad AR: raíces fuera del círculo unitario",
            float(np.min(np.abs(raices_ar))),
            bool(np.all(np.abs(raices_ar) > 1))
        ))

    return pd.DataFrame(filas, columns=["Criterio", "Valor", "Cumple"])



def revisar_significancia_parametros(res, alpha=0.05):
    tabla = pd.DataFrame({
        "estimación": res.params,
        "p-valor": res.pvalues
    })

    tabla["significativo"] = tabla["p-valor"] < alpha
    tabla.index.name = "Parámetro"
    return tabla


def metricas_modelo(res):
    """Resume ajuste, parsimonia y convergencia del modelo estimado."""
    flag = getattr(res, "convergence_flag", None)
    return pd.Series({
        "log-verosimilitud": res.loglikelihood,
        "AIC": res.aic,
        "BIC": res.bic,
        "observaciones": getattr(res, "nobs", np.nan),
        "número_parámetros": len(res.params),
        "convergió": (flag == 0) if flag is not None else np.nan,
    }, name="métrica")

def prueba_sign_bias(res):
    z = obtener_residuos_estandarizados(res)
    z_lag = z.shift(1)

    s_neg = (z_lag < 0).astype(float)
    s_pos = 1 - s_neg

    datos = pd.DataFrame({
        "z2": z**2,
        "s_neg": s_neg,
        "s_neg_z": s_neg * z_lag,
        "s_pos_z": s_pos * z_lag
    }).dropna()

    X = sm.add_constant(datos[["s_neg", "s_neg_z", "s_pos_z"]])
    ajuste = sm.OLS(datos["z2"], X).fit()

    prueba = ajuste.f_test(
        "s_neg = s_neg_z = s_pos_z = 0"
    )

    return {
        "estadístico F": float(prueba.statistic),
        "p-valor": float(prueba.pvalue)
    }

def pronostico_egarch(r, train_fin, obj_ini, obj_fin, espec=None):
    if espec is None:
        espec = {
            "mean": "Constant",
            "vol": "EGARCH",
            "p": 1,
            "o": 1,
            "q": 1,
            "dist": "t",
        }

    # Estima el modelo solo con los datos de entrenamiento del fold
    res_tr = arch_model(r.loc[:train_fin], **espec).fit(disp="off")

    # Pronostica la varianza a un paso para el periodo objetivo
    r_full = r.loc[:obj_fin]
    pos_ini = r_full.index.searchsorted(pd.Timestamp(obj_ini))
    origen = pos_ini - 1

    fc = arch_model(r_full, **espec).forecast(
        params=res_tr.params,
        horizon=1,
        start=origen,
        reindex=False
    )

    h = fc.variance["h.1"].to_numpy()
    objetivos = r_full.index[origen + 1:]

    h = pd.Series(h[:len(objetivos)], index=objetivos, name="h_egarch")
    return h.loc[obj_ini:obj_fin], res_tr


def perdidas(y, h):
    return pd.DataFrame({"mse": (y - h)**2, "qlike": y / h + np.log(h)})

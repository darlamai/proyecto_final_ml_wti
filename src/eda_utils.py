"""Funciones del EDA (tablas y gráficos). Todas reciben los datos como argumento: nada global."""
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from arch.unitroot import PhillipsPerron
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tsa.stattools import adfuller, acf, grangercausalitytests, kpss
from . import config as cfg

warnings.filterwarnings("ignore")




def chequeo_calidad(base):
    """Mínimo, valores <= 0, infinitos, nulos y precios repetidos (posible relleno) por serie."""
    return pd.DataFrame({
        "min": base.min(),
        "n_<=0": (base <= 0).sum(),
        "n_inf": np.isinf(base).sum(),
        "n_nulos": base.isna().sum(),
        "n_dias_sin_cambio": (base.diff() == 0).sum(),
    })




def saltos_extremos(base, top=5):
    """
    Devuelve las mayores variaciones diarias por serie con fechas y precios.

    Si ambos precios son positivos, calcula también el cambio logarítmico.
    Si alguno es cero o negativo, no calcula el log y devuelve NaN en
    ``log_var_abs``. El cambio absoluto y porcentual sí se calculan.
    """
    filas = []

    for c in base.columns:
        precio_actual = base[c]
        precio_previo = precio_actual.shift(1)
        cambio = precio_actual - precio_previo

        cambio_log = pd.Series(np.nan, index=base.index, dtype=float)
        pares_positivos = (precio_actual > 0) & (precio_previo > 0)
        cambio_log.loc[pares_positivos] = (
            np.log(precio_actual.loc[pares_positivos])
            - np.log(precio_previo.loc[pares_positivos])
        )

        # Ordena por cambio logarítmico cuando está definido;
        # los casos con cero o negativos se ordenan por cambio absoluto.
        puntaje = cambio_log.abs().fillna(cambio.abs())
        fechas = puntaje.nlargest(top).index

        for fecha in fechas:
            anterior = precio_previo.loc[fecha]
            actual = precio_actual.loc[fecha]

            if pd.isna(anterior) or pd.isna(actual):
                continue

            filas.append({
                "serie": c,
                "fecha_previa": base.index[base.index.get_loc(fecha) - 1].date(),
                "fecha": fecha.date(),
                "precio_previo": anterior,
                "precio_actual": actual,
                "cambio": actual - anterior,
                "cambio_pct": (
                    (actual / anterior - 1) * 100
                    if anterior != 0 else np.nan
                ),
                "log_var_abs": abs(cambio_log.loc[fecha]),
            })

    return pd.DataFrame(filas).sort_values(
        "log_var_abs",
        ascending=False,
        na_position="last",
        ignore_index=True,
    )


def dias_sin_cambio(base):
    """
    Identifica las fechas en que una serie tiene el mismo valor
    que en la observación anterior.

    La comparación se realiza entre filas consecutivas del DataFrame,
    que normalmente corresponden a sesiones de negociación consecutivas.

    Parameters
    ----------
    base : pd.DataFrame
        DataFrame con índice de fechas y una o más series numéricas.

    Returns
    -------
    pd.DataFrame
        Tabla con la serie, la fecha actual, la fecha previa
        y el valor que se mantuvo sin cambio.
    """
    filas = []

    for columna in base.columns:
        sin_cambio = base[columna].eq(base[columna].shift(1))

        for fecha in base.index[sin_cambio]:
            posicion = base.index.get_loc(fecha)

            filas.append({
                "serie": columna,
                "fecha_previa": base.index[posicion - 1].date(),
                "fecha": fecha.date(),
                "valor": base[columna].loc[fecha],
            })

    return pd.DataFrame(
        filas,
        columns=["serie", "fecha_previa", "fecha", "valor"],
    )


def top_movimientos_diarios(base, columna="WTI", top=10):
    """
    Devuelve los mayores movimientos diarios de una serie de precios.

    El ranking usa el valor absoluto del retorno logarítmico, calculado
    solo cuando el precio actual y el anterior son positivos. Los cambios
    que involucran precios cero o negativos se devuelven en una tabla aparte.

    Parameters
    ----------
    base : pd.DataFrame
        DataFrame con índice de fechas y la serie de precios.
    columna : str, default="WTI"
        Nombre de la columna que se analizará.
    top : int, default=10
        Número de mayores movimientos logarítmicos que se mostrarán.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        Tabla de mayores movimientos y tabla de cambios excluidos del log.
    """
    s = base[columna]
    prev_close = s.shift(1)

    # El retorno logarítmico requiere precios actual y previo positivos.
    pares_positivos = s.gt(0) & prev_close.gt(0)
    r = pd.Series(float("nan"), index=s.index, dtype=float)
    r.loc[pares_positivos] = (
        np.log(s.loc[pares_positivos] / prev_close.loc[pares_positivos]) * 100
    )

    # Retorno simple porcentual.
    r_simple = (s / prev_close - 1) * 100

    fechas_top = r.abs().nlargest(top).index
    tabla_top = pd.DataFrame({
        "día": fechas_top.day_name(),
        "cierre_anterior": prev_close.loc[fechas_top],
        "cierre": s.loc[fechas_top],
        "retorno_log_%": r.loc[fechas_top],
        "retorno_simple_%": r_simple.loc[fechas_top],
    }).sort_index()

    tabla_top.index = tabla_top.index.date

    # Cambios que no admiten retorno logarítmico.
    excluir_log = s.le(0) | prev_close.le(0)
    fechas_excluidas = s.index[excluir_log]

    tabla_excluidos = pd.DataFrame({
        "cierre_anterior": prev_close.loc[fechas_excluidas],
        "cierre": s.loc[fechas_excluidas],
        "cambio_USD": (s - prev_close).loc[fechas_excluidas],
        "retorno_simple_%": r_simple.loc[fechas_excluidas],
    })
    tabla_excluidos.index.name = "fecha"

    return tabla_top.round(2), tabla_excluidos.round(2)


def estadisticos_por_anio(px, c):
    """Conteo, media, desviación, mínimo, mediana y máximo del nivel por año."""
    return px[c].groupby(px.index.year).agg(["count", "mean", "std", "min", "median", "max"]).round(2)


# ---- EDA descriptivo de los precios (usar SOLO con precios de train) ------------------
def descriptivos_precios(px):
    """Descriptivos de los niveles (precios) de cada serie."""
    filas = []
    for c in px.columns:
        s = px[c]
        jb_stat, jb_p = stats.jarque_bera(s)
        filas.append({
            "serie": c, "n": len(s), "media": s.mean(), "std": s.std(), "CV": s.std() / s.mean(),
            "min": s.min(), "fecha_min": s.idxmin().date(),
            "p25": s.quantile(0.25), "mediana": s.median(), "p75": s.quantile(0.75),
            "max": s.max(), "fecha_max": s.idxmax().date(),
            "asimetria": stats.skew(s), "curtosis_exceso": stats.kurtosis(s),
            "Jarque_Bera_stat": jb_stat,
            "Jarque_Bera_p": jb_p
        })
    return pd.DataFrame(filas).set_index("serie")



def graficar_serie_temporal(serie, nombre=None, unidad="Precio"):
    """
    Grafica una serie temporal.

    Parameters
    ----------
    serie : pd.Series
        Serie numérica con las fechas en el índice.
    nombre : str, optional
        Nombre para el título y la leyenda. Si no se indica, usa serie.name.
    unidad : str, default="Precio"
        Etiqueta del eje vertical.
    """
    nombre = nombre or serie.name or "Serie"

    fig, ax = plt.subplots(figsize=(13, 4))
    ax.plot(serie.index, serie.values, linewidth=0.8, label=nombre)
    ax.set_title(f"Serie temporal de {nombre}")
    ax.set_xlabel("Fecha")
    ax.set_ylabel(unidad)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    plt.show()


def drop_nonpositive_rows(base, columns=None):
    """
    Elimina filas con valores menores o iguales a cero en columnas numéricas.

    Parameters
    ----------
    base : pd.DataFrame
        DataFrame que se desea limpiar.
    columns : list[str], optional
        Columnas que se revisarán. Si es None, revisa todas las numéricas.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        DataFrame limpio y filas eliminadas.
    """
    if columns is None:
        columnas_revision = base.select_dtypes(include="number").columns
    else:
        columnas_revision = columns

    mask_no_positivos = base[columnas_revision].le(0).any(axis=1)

    base_limpia = base.loc[~mask_no_positivos].copy()
    filas_eliminadas = base.loc[mask_no_positivos].copy()

    return base_limpia, filas_eliminadas


CANDIDATAS = {
    "nivel":  lambda s: s,
    "diff":   lambda s: s.diff(),
    "logret": lambda s: np.log(s).diff(),
}

def unidad(c):
    return "puntos" if cfg.transf[c] == "diff" else "log-retorno"


def observaciones_por_anio(base):
    return base.groupby(base.index.year).size().rename("n_obs").to_frame()


#### TIME SERIES CROSS VALIDATION################################################

def definir_folds(n_folds=5, repeticiones=((2020, 1), (2019, 7)), inicio=cfg.DEV_INI):
    # Cada repetición: (año, mes) en que empieza la primera validación
    filas = []
    for rep, (anio0, mes0) in enumerate(repeticiones, start=1):
        for k in range(n_folds):
            val_ini = pd.Timestamp(anio0 + k, mes0, 1)
            val_fin = val_ini + pd.DateOffset(years=1) - pd.Timedelta(days=1)
            filas.append({"rep": rep, "fold": k + 1,
                          "train_ini": pd.Timestamp(inicio),
                          "train_fin": val_ini - pd.Timedelta(days=1),
                          "val_ini": val_ini, "val_fin": val_fin})
    return pd.DataFrame(filas)

def indices_fold(idx, fila):
    # Posiciones de train y validación para cualquier índice de fechas (base_ret, X, etc.)
    tr = np.where((idx >= fila.train_ini) & (idx <= fila.train_fin))[0]
    va = np.where((idx >= fila.val_ini) & (idx <= fila.val_fin))[0]
    return tr, va


def graficar_niveles_con_cortes(base, columnas=None, splits=None, mostrar_eda=True):
    """Niveles de cada serie con el corte desarrollo (train+val) / test final.

    splits: dict con "dev" y "test" como (inicio, fin) y, opcional, "eda_fin".
    """
    columnas = columnas or list(base.columns)
    splits = splits or cfg.splits
    dev_fin = pd.Timestamp(splits["dev"][1])
    test_ini, test_fin = map(pd.Timestamp, splits["test"])
    eda_fin = pd.Timestamp(splits["eda_fin"]) if mostrar_eda and "eda_fin" in splits else None

    fig, axes = plt.subplots(len(columnas), 1, figsize=(13, 2.3 * len(columnas)), sharex=True)
    for ax, c in zip(np.atleast_1d(axes), columnas):
        ax.plot(base.index, base[c], lw=0.7)
        ax.axvspan(test_ini, min(test_fin, base.index.max()), color="red", alpha=0.08)
        ax.axvline(dev_fin, color="red", ls="--", lw=1)
        if eda_fin is not None:
            ax.axvline(eda_fin, color="grey", ls=":", lw=1)
        ax.set_ylabel(c); ax.grid(alpha=0.3)

    titulo = "Niveles: desarrollo (train + validación) | test final (sombreado rojo)"
    if eda_fin is not None:
        titulo += f"\nlínea gris: fin del tramo de diagnóstico ({eda_fin.date()})"
    np.atleast_1d(axes)[0].set_title(titulo)
    plt.tight_layout(); plt.show()


def diagnostico_estacionariedad(ret):
    try:
        from arch.unitroot import PhillipsPerron
    except ImportError as exc:
        raise ImportError(
            "La prueba Phillips-Perron requiere el paquete 'arch'. "
            "Instálalo en el entorno del notebook con: %pip install arch"
        ) from exc

    filas = []

    for c in ret.columns:
        s = ret[c].dropna()

        adf_stat, adf_p = adfuller(
            s, autolag="AIC"
        )[:2]

        kpss_stat, kpss_p = kpss(
            s, regression="c", nlags="auto"
        )[:2]

        pp = PhillipsPerron(s, trend="c")

        filas.append({
            "serie": c,
            "n": len(s),
            "ADF_stat": adf_stat,
            "ADF_p": adf_p,
            "KPSS_stat": kpss_stat,
            "KPSS_p": kpss_p,
            "PP_stat": pp.stat,
            "PP_p": pp.pvalue,
            "PP_lags": pp.lags,
        })

    return pd.DataFrame(filas).round(4)




def diagnostico_estacionalidad(ret, rezagos=(5, 21, 252)):
    filas = []

    for c in ret.columns:
        s = ret[c].dropna()

        grupos = [
            s[s.index.month == mes].values
            for mes in range(1, 13)
        ]

        if all(len(g) >= 5 for g in grupos):
            kw_stat, kw_p = stats.kruskal(*grupos)
        else:
            kw_stat, kw_p = np.nan, np.nan

        max_lag = min(max(rezagos), len(s) - 1)

        acf_vals = acf(
            s,
            nlags=max_lag,
            fft=True
        )

        fila = {
            "serie": c,
            "n": len(s),
            "Kruskal_mes_p": kw_p
        }

        for lag in rezagos:
            fila[f"ACF_{lag}"] = (
                acf_vals[lag]
                if lag <= max_lag
                else np.nan
            )

        filas.append(fila)

    return pd.DataFrame(filas).round(4)




def graficar_asimetria_curtosis(datos):
    """Grafica la asimetría y la curtosis de una serie o de cada columna.

    La curtosis mostrada es el exceso de curtosis: una distribución normal
    tiene valor 0. Se ignoran valores nulos e infinitos.
    """
    if isinstance(datos, pd.Series):
        datos = datos.to_frame(name=datos.name or "serie")
    columnas = datos.select_dtypes(include="number").columns
    if len(columnas) == 0:
        raise ValueError("Se necesita al menos una columna numérica.")

    asimetria, curtosis = {}, {}
    for columna in columnas:
        valores = datos[columna].replace([np.inf, -np.inf], np.nan).dropna()
        if valores.empty:
            asimetria[columna] = np.nan
            curtosis[columna] = np.nan
        else:
            asimetria[columna] = stats.skew(valores)
            curtosis[columna] = stats.kurtosis(valores)  # exceso (Fisher)

    fig, axes = plt.subplots(1, 2, figsize=(12, max(4, 0.45 * len(columnas))))
    pd.Series(asimetria).sort_values().plot.barh(ax=axes[0], color="steelblue")
    axes[0].axvline(0, color="black", lw=0.8)
    axes[0].set_title("Asimetría")
    axes[0].set_xlabel("Asimetría de Fisher")
    pd.Series(curtosis).sort_values().plot.barh(ax=axes[1], color="darkorange")
    axes[1].axvline(0, color="black", lw=0.8, label="Normal = 0")
    axes[1].set_title("Curtosis (exceso)")
    axes[1].set_xlabel("Exceso de curtosis")
    axes[1].legend()
    fig.tight_layout()
    plt.show()
    return fig, axes


def graficar_histograma(serie, bins=30, nombre=None):
    """Grafica el histograma de una Serie o de cada columna numérica de un DataFrame."""
    es_serie = isinstance(serie, pd.Series)
    datos = serie.to_frame(name=serie.name or "Serie") if es_serie else serie
    if not isinstance(datos, pd.DataFrame):
        datos = pd.Series(datos, name=nombre or "Serie").to_frame()
    columnas = datos.select_dtypes(include="number").columns
    if len(columnas) == 0:
        raise ValueError("Se necesita al menos una columna numérica.")

    ncols = 1 if es_serie else min(3, len(columnas))
    nrows = int(np.ceil(len(columnas) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 3.8 * nrows), squeeze=False)
    for ax, columna in zip(axes.ravel(), columnas):
        valores = datos[columna].replace([np.inf, -np.inf], np.nan).dropna()
        if valores.empty:
            ax.set_title(f"{columna}: sin valores válidos")
            ax.axis("off")
            continue
        etiqueta = nombre or str(columna)
        ax.hist(valores, bins=bins, edgecolor="white", alpha=0.8)
        ax.set(title=f"Histograma de {etiqueta}", xlabel=etiqueta, ylabel="Frecuencia")
        ax.grid(axis="y", alpha=0.25)
    for ax in axes.ravel()[len(columnas):]:
        ax.axis("off")
    fig.tight_layout()
    plt.show()
    return fig, axes.ravel()[0] if es_serie else axes


def graficar_qqplot(serie, nombre=None):
    """Compara con una normal los cuantiles de una Serie o columnas numéricas."""
    es_serie = isinstance(serie, pd.Series)
    datos = serie.to_frame(name=serie.name or "Serie") if es_serie else serie
    if not isinstance(datos, pd.DataFrame):
        datos = pd.Series(datos, name=nombre or "Serie").to_frame()
    columnas = datos.select_dtypes(include="number").columns
    if len(columnas) == 0:
        raise ValueError("Se necesita al menos una columna numérica.")

    ncols = 1 if es_serie else min(3, len(columnas))
    nrows = int(np.ceil(len(columnas) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 5 * nrows), squeeze=False)
    for ax, columna in zip(axes.ravel(), columnas):
        valores = datos[columna].replace([np.inf, -np.inf], np.nan).dropna()
        titulo = nombre or str(columna)
        if len(valores) < 2:
            ax.set_title(f"{titulo}: menos de dos valores válidos")
            ax.axis("off")
            continue
        sm.qqplot(valores, line="45", ax=ax, fit=True)
        ax.set_title(f"QQ-plot normal de {titulo}")
    for ax in axes.ravel()[len(columnas):]:
        ax.axis("off")
    fig.tight_layout()
    plt.show()
    return fig, axes.ravel()[0] if es_serie else axes


def graficar_boxplot_anual(px):
    """Boxplot por año de cada serie: cambios de nivel y de dispersión entre años."""
    cols = list(px.columns)
    anios = sorted(px.index.year.unique())
    fig, axes = plt.subplots(len(cols), 1, figsize=(12, 2.4 * len(cols)), sharex=True)
    for ax, c in zip(np.atleast_1d(axes), cols):
        ax.boxplot([px.loc[px.index.year == a, c].values for a in anios])
        ax.set_ylabel(c); ax.grid(alpha=0.3)
        ax.set_xticks(range(1, len(anios) + 1)); ax.set_xticklabels(anios)
    plt.tight_layout(); plt.show()




def graficar_acf_con_pvalores(ret, lags=40, rezagos_prueba=(5, 21, 252), etiqueta="r_t"):
    """Grafica ACF de r_t y calcula pruebas Ljung–Box para varios rezagos.

    Ljung–Box contrasta conjuntamente H0: las autocorrelaciones hasta el
    rezago indicado son cero. Devuelve (figura, ejes, tabla de resultados).
    """
    es_serie = isinstance(ret, pd.Series)
    datos = ret.to_frame(name=ret.name or "serie") if es_serie else ret
    if not isinstance(datos, pd.DataFrame):
        datos = pd.DataFrame(datos)
    columnas = datos.select_dtypes(include="number").columns
    if len(columnas) == 0:
        raise ValueError("Se necesita al menos una columna numérica.")

    rezagos_prueba = sorted(set(int(k) for k in rezagos_prueba if int(k) > 0))
    if not rezagos_prueba:
        raise ValueError("rezagos_prueba debe incluir al menos un entero positivo.")

    fig, axes = plt.subplots(len(columnas), 1, figsize=(10, 3.5 * len(columnas)), squeeze=False)
    filas = []
    for ax, columna in zip(axes.ravel(), columnas):
        r = datos[columna].replace([np.inf, -np.inf], np.nan).dropna()
        if len(r) < 3:
            raise ValueError(f"La columna {columna!r} tiene menos de 3 observaciones válidas.")
        max_lag = min(int(lags), len(r) // 2)
        sm.graphics.tsa.plot_acf(
            r, lags=max_lag, ax=ax, zero=False, alpha=0.05,
            bartlett_confint=False, title=f"{columna}: ACF de {etiqueta}",
        )
        ax.set_xlabel("Rezago")
        ax.set_ylabel("Autocorrelación")
        ax.grid(alpha=0.2)

        validos = [k for k in rezagos_prueba if k < len(r)]
        if not validos:
            raise ValueError(f"Ningún rezago de prueba es menor que el tamaño de {columna!r}.")
        prueba = acorr_ljungbox(r, lags=validos, return_df=True)
        fila = {"serie": columna, "n": len(r)}
        for k in validos:
            fila[f"LB_Q({k})"] = prueba.loc[k, "lb_stat"]
            fila[f"LB_p({k})"] = prueba.loc[k, "lb_pvalue"]
        filas.append(fila)

    fig.tight_layout()
    plt.show()
    resultados = pd.DataFrame(filas).set_index("serie").round(4)
    return fig, axes.ravel(), resultados


def graficar_pacf_con_pvalores(ret, lags=40):
    """Grafica la PACF y devuelve sus coeficientes y p-valores aproximados.

    Los p-valores usan la aproximación normal con error estándar 1/sqrt(n);
    evalúan cada rezago por separado, no de forma conjunta.
    """
    es_serie = isinstance(ret, pd.Series)
    datos = ret.to_frame(name=ret.name or "serie") if es_serie else ret
    if not isinstance(datos, pd.DataFrame):
        datos = pd.DataFrame(datos)
    columnas = datos.select_dtypes(include="number").columns
    if len(columnas) == 0:
        raise ValueError("Se necesita al menos una columna numérica.")

    fig, axes = plt.subplots(len(columnas), 1, figsize=(10, 3.5 * len(columnas)), squeeze=False)
    filas = []
    for ax, columna in zip(axes.ravel(), columnas):
        r = datos[columna].replace([np.inf, -np.inf], np.nan).dropna()
        if len(r) < 3:
            raise ValueError(f"La columna {columna!r} tiene menos de 3 observaciones válidas.")
        max_lag = min(int(lags), len(r) // 2 - 1)
        if max_lag < 1:
            raise ValueError(f"No hay suficientes observaciones para calcular la PACF de {columna!r}.")

        coeficientes = sm.tsa.stattools.pacf(r, nlags=max_lag, method="ywm")
        sm.graphics.tsa.plot_pacf(
            r, lags=max_lag, ax=ax, zero=False, alpha=0.05, method="ywm",
            title=f"{columna}: PACF de retornos r_t",
        )
        ax.set_xlabel("Rezago")
        ax.set_ylabel("Autocorrelación parcial")
        ax.grid(alpha=0.2)

        se = 1 / np.sqrt(len(r))
        for k, coef in enumerate(coeficientes[1:], start=1):
            filas.append({
                "serie": columna,
                "rezago": k,
                "PACF": coef,
                "p_aprox": 2 * stats.norm.sf(abs(coef / se)),
            })

    fig.tight_layout()
    plt.show()
    resultados = pd.DataFrame(filas).set_index(["serie", "rezago"]).round(4)
    return fig, axes.ravel(), resultados




def graficar_volatilidad_movil(ret, ventanas=(21, 63)):
    """Grafica desviación estándar móvil de una Serie o de columnas numéricas."""
    datos = ret.to_frame(name=ret.name or "serie") if isinstance(ret, pd.Series) else ret
    if not isinstance(datos, pd.DataFrame):
        datos = pd.DataFrame(datos)
    columnas = datos.select_dtypes(include="number").columns
    ventanas = tuple(int(w) for w in ventanas if int(w) > 1)
    if len(columnas) == 0 or not ventanas:
        raise ValueError("Se requieren columnas numéricas y ventanas mayores que 1.")

    fig, axes = plt.subplots(len(columnas), 1, figsize=(12, 3 * len(columnas)), squeeze=False)
    for ax, columna in zip(axes.ravel(), columnas):
        s = datos[columna].replace([np.inf, -np.inf], np.nan)
        for w in ventanas:
            ax.plot(s.rolling(w).std(), lw=1, label=f"Ventana {w}")
        ax.set_title(f"{columna}: volatilidad móvil")
        ax.set_ylabel("Desviación estándar")
        ax.grid(alpha=0.25)
        ax.legend()
    fig.tight_layout()
    plt.show()
    return fig, axes.ravel()


def diagnostico_arch_lm(residuos_media, lags=10):
    """Aplica ARCH-LM a residuos de una ecuación de media, por serie.

    H0: no hay efectos ARCH hasta el número de rezagos especificado.
    """
    datos = (residuos_media.to_frame(name=residuos_media.name or "serie")
             if isinstance(residuos_media, pd.Series) else residuos_media)
    if not isinstance(datos, pd.DataFrame):
        datos = pd.DataFrame(datos)
    columnas = datos.select_dtypes(include="number").columns
    if len(columnas) == 0:
        raise ValueError("Se necesita al menos una columna numérica de residuos.")
    if int(lags) < 1:
        raise ValueError("lags debe ser un entero positivo.")

    filas = []
    for columna in columnas:
        s = datos[columna].replace([np.inf, -np.inf], np.nan).dropna()
        if len(s) <= int(lags) + 1:
            raise ValueError(f"No hay suficientes residuos válidos en {columna!r} para lags={lags}.")
        lm_stat, lm_p, f_stat, f_p = het_arch(s, nlags=int(lags))
        filas.append({
            "serie": columna,
            "n": len(s),
            "ARCH_LM_stat": lm_stat,
            "ARCH_LM_p": lm_p,
            "ARCH_F_stat": f_stat,
            "ARCH_F_p": f_p,
            "lags": int(lags),
        })
    return pd.DataFrame(filas).set_index("serie").round(4)


def outliers_z(r, k=None):
    """Observaciones con |z| > k, con media y desviación de la propia serie que se pase (train)."""
    k = cfg.umbral_outlier_z if k is None else k
    r = r.dropna()
    return r[(r - r.mean()).abs() > k * r.std()]






















































































































































































































































































































































































"""Funciones del EDA (tablas y gráficos). Todas reciben los datos como argumento: nada global."""
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tsa.stattools import adfuller, grangercausalitytests, kpss

from . import config as cfg

warnings.filterwarnings("ignore")


def unidad(c):
    return "puntos" if cfg.transf[c] == "diff" else "log-retorno"


# =====================================================================================
# 01 · Datos crudos (precios)
# =====================================================================================
def graficar_niveles_con_cortes(base, columnas=None, splits=None):
    """Niveles de cada serie con líneas verticales en los cortes train / val / test."""
    columnas = columnas or list(base.columns)
    splits = splits or cfg.splits
    cortes = [pd.Timestamp(splits["train"][1]), pd.Timestamp(splits["val"][1])]
    fig, axes = plt.subplots(len(columnas), 1, figsize=(13, 2.3 * len(columnas)), sharex=True)
    for ax, c in zip(np.atleast_1d(axes), columnas):
        ax.plot(base.index, base[c], lw=0.7)
        ax.axvline(cortes[0], color="red", ls="--", lw=1)
        ax.axvline(cortes[1], color="orange", ls="--", lw=1)
        ax.set_ylabel(c); ax.grid(alpha=0.3)
    np.atleast_1d(axes)[0].set_title("Niveles (rojo: fin de train, naranja: fin de validación)")
    plt.tight_layout(); plt.show()


def chequeo_calidad(base):
    """Mínimo, valores <= 0, infinitos, nulos y precios repetidos (posible relleno) por serie."""
    return pd.DataFrame({
        "min": base.min(),
        "n_<=0": (base <= 0).sum(),
        "n_inf": np.isinf(base).sum(),
        "n_nulos": base.isna().sum(),
        "n_dias_sin_cambio": (base.diff() == 0).sum(),
    })


def observaciones_por_anio(base):
    return base.groupby(base.index.year).size().rename("n_obs").to_frame()


def saltos_extremos(base, top=5):
    """
    Mayores variaciones diarias en valor absoluto, para revisar a mano si son errores de datos
    o cambios de contrato. Es una auditoría de calidad: NO se usa para decidir el diseño.
    """
    filas = []
    for c in base.columns:
        s = base[c].where(base[c] > 0)
        v = np.log(s).diff().abs().nlargest(top)
        for fecha, val in v.items():
            filas.append({"serie": c, "fecha": fecha.date(), "|log-var|": round(val, 4)})
    return pd.DataFrame(filas)


# ---- EDA descriptivo de los precios (usar SOLO con precios de train) ------------------
def descriptivos_precios(px):
    """Descriptivos de los niveles (precios) de cada serie."""
    filas = []
    for c in px.columns:
        s = px[c]
        filas.append({
            "serie": c, "n": len(s), "media": s.mean(), "std": s.std(), "CV": s.std() / s.mean(),
            "min": s.min(), "fecha_min": s.idxmin().date(),
            "p25": s.quantile(0.25), "mediana": s.median(), "p75": s.quantile(0.75),
            "max": s.max(), "fecha_max": s.idxmax().date(),
            "asimetria": stats.skew(s), "curtosis_exceso": stats.kurtosis(s),
        })
    return pd.DataFrame(filas).set_index("serie")


def graficar_base100(px):
    """Evolución relativa de todas las series (base 100 = primer día, escala logarítmica)."""
    (px / px.iloc[0] * 100).plot(figsize=(13, 4.5), lw=0.9, logy=True, grid=True)
    plt.title("Evolución relativa (base 100 = primer día de train, escala log)")
    plt.tight_layout(); plt.show()


def graficar_medias_moviles(px, w1=63, w2=252):
    """Cada serie con su media móvil trimestral (w1) y anual (w2): tendencias y cambios de régimen."""
    cols = list(px.columns)
    fig, axes = plt.subplots(len(cols), 1, figsize=(13, 2.3 * len(cols)), sharex=True)
    for ax, c in zip(np.atleast_1d(axes), cols):
        ax.plot(px[c], lw=0.5, color="0.6", label=c)
        ax.plot(px[c].rolling(w1).mean(), lw=1.0, label=f"media {w1} d")
        ax.plot(px[c].rolling(w2).mean(), lw=1.4, label=f"media {w2} d")
        ax.set_ylabel(c); ax.grid(alpha=0.3); ax.legend(loc="upper left", ncol=3, fontsize=7)
    plt.tight_layout(); plt.show()


def graficar_histogramas_niveles(px):
    """Distribución de los niveles de cada serie."""
    cols = list(px.columns)
    fig, axes = plt.subplots(2, 4, figsize=(15, 6))
    for ax, c in zip(axes.ravel(), cols):
        ax.hist(px[c], bins=50, alpha=0.7); ax.set_title(c)
    for ax in axes.ravel()[len(cols):]:
        ax.axis("off")
    plt.tight_layout(); plt.show()


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


def estadisticos_por_anio(px, c):
    """Conteo, media, desviación, mínimo, mediana y máximo del nivel por año."""
    return px[c].groupby(px.index.year).agg(["count", "mean", "std", "min", "median", "max"]).round(2)


def drawdown_maximo(s):
    """Mayor caída desde un máximo previo hasta el valle posterior."""
    dd = s / s.cummax() - 1
    valle = dd.idxmin()
    return {"pico": s.loc[:valle].idxmax().date(), "valle": valle.date(), "caida_%": round(100 * dd.min(), 1)}


def tabla_drawdown(px):
    return pd.DataFrame({c: drawdown_maximo(px[c]) for c in px.columns}).T


def graficar_estacionalidad_nivel(px, c):
    """
    Índice estacional del nivel: precio / media del año, promediado por mes.
    Dividir entre la media anual quita el efecto de la tendencia entre años. 1 = el promedio del año.
    """
    idx = px[c] / px[c].groupby(px.index.year).transform("mean")
    idx.groupby(idx.index.month).mean().plot.bar(figsize=(8, 3.2), title=f"{c}: índice estacional del nivel por mes")
    plt.axhline(1, color="k", ls=":", lw=0.8)
    plt.tight_layout(); plt.show()


def graficar_correlacion_niveles(px):
    """Spearman entre NIVELES. Es espuria (tendencias): solo ilustra por qué se transforma. No decide nada."""
    corr = px.corr(method="spearman")
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr))); ax.set_xticklabels(corr.columns, rotation=45)
    ax.set_yticks(range(len(corr))); ax.set_yticklabels(corr.columns)
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
    plt.colorbar(im); ax.set_title("Spearman de niveles (espuria, solo ilustrativa)")
    plt.tight_layout(); plt.show()


CANDIDATAS = {
    "nivel":  lambda s: s,
    "diff":   lambda s: s.diff(),
    "logret": lambda s: np.log(s).diff(),
}


def diagnostico_transformaciones(px):
    """
    Compara nivel / diff / logret de cada serie. Usar SOLO con precios de train.
    Métricas de escala: corr_abs_nivel, spearman_abs_nivel, ratio_std_alto_bajo.
    """
    filas = []
    for c in px.columns:
        for nombre, f in CANDIDATAS.items():
            s = f(px[c]).dropna()
            x = px[c].reindex(s.index)                       # nivel alineado con s
            adf_stat, adf_p = adfuller(s, autolag="AIC")[:2]
            if nombre != "nivel":
                terc = pd.qcut(x, 3, labels=False, duplicates="drop")
                sd_t = s.groupby(terc).std()
                ratio = sd_t.iloc[-1] / sd_t.iloc[0]
            else:
                ratio = np.nan                               # sin sentido en el nivel
            filas.append({
                "serie": c, "transf": nombre, "n": len(s),
                "ADF_p": adf_p,          # < 0.05: estacionaria
                "KPSS_p": kpss(s, regression="c", nlags="auto")[1],   # > 0.05: estacionaria (tope 0.10)
                "asimetria": stats.skew(s),
                "curtosis": stats.kurtosis(s),          # en exceso: > 0 = colas más pesadas que la normal
                # corr_abs_nivel: corr(|variación_t|, nivel_t). ≈ 0 (entre -0.1 y 0.1): escala adecuada;
                # > 0.1: movimientos mayores con precio alto; < -0.1: mayores con precio bajo.
                "corr_abs_nivel": s.abs().corr(x),
                "ADF_stat": adf_stat,                   # solo para comparar; decide el p-valor
                "spearman_abs_nivel": stats.spearmanr(s.abs(), x)[0],   # igual, robusta a outliers
                # ratio_std_alto_bajo: desviación en el tercil alto del nivel / en el bajo. ≈ 1: estable.
                "ratio_std_alto_bajo": ratio,
            })
    return pd.DataFrame(filas).round(4)


def graficar_transformaciones(px, c, lags=40):
    """Nivel / diff / logret de una serie: serie, histograma y dependencia respecto al nivel."""
    p = px[c]
    fig, ax = plt.subplots(3, 3, figsize=(16, 10))
    for j, (nombre, f) in enumerate(CANDIDATAS.items()):
        s = f(p).dropna()
        ax[0, j].plot(s, lw=0.6)
        ax[0, j].axhline(s.mean(), color="red", ls="--", lw=0.8)
        ax[0, j].set_title(f"{c}: {nombre}")
        ax[1, j].hist(s, bins=80, density=True, alpha=0.6)
        xs = np.linspace(s.min(), s.max(), 400)
        ax[1, j].plot(xs, stats.norm.pdf(xs, s.mean(), s.std()), "r", lw=1)
        ax[1, j].set_title(f"asim={stats.skew(s):.2f}  curt={stats.kurtosis(s):.2f}")
        if nombre == "nivel":
            sm.graphics.tsa.plot_acf(s, lags=lags, ax=ax[2, j])
            ax[2, j].set_title("ACF del nivel")
        else:
            x = p.reindex(s.index)
            ax[2, j].scatter(x, s.abs(), s=3, alpha=0.3)
            xo = np.sort(x.values)
            ax[2, j].plot(xo, np.polyval(np.polyfit(x, s.abs(), 1), xo), "r")
            ax[2, j].set_xlabel("nivel"); ax[2, j].set_ylabel("|x|")
            ax[2, j].set_title(f"|{nombre}| vs nivel (r={s.abs().corr(x):.2f})")
    plt.tight_layout(); plt.show()


def graficar_vol(px, c, w=63):
    """Estabilidad de la variación: diff vs logret, normalizadas para ser comparables."""
    p = px[c]
    fig, ax = plt.subplots(1, 2, figsize=(15, 3.5))
    anual = {}
    for nombre in ["diff", "logret"]:
        s = CANDIDATAS[nombre](p).dropna()
        (s.rolling(w).std() / s.std()).plot(ax=ax[0], label=nombre, lw=0.9)
        sd = s.groupby(s.index.year).std()
        anual[nombre] = sd / sd.mean()
    ax[0].axhline(1, color="k", ls=":", lw=0.8); ax[0].legend()
    ax[0].set_title(f"{c}: desviación móvil {w} d / desviación total")
    pd.DataFrame(anual).plot.bar(ax=ax[1])
    ax[1].set_title("Desviación anual / promedio de las desviaciones anuales")
    plt.tight_layout(); plt.show()


def elasticidad_vol_nivel(px, series, ventanas=(21, 63, 126)):
    """
    k en  log(desv. móvil de la diferencia) = a + k·log(nivel medio móvil).
    k ≈ 0: la variación en puntos es constante (conviene diff); k ≈ 1: proporcional al nivel (logret).
    """
    filas = []
    for c in series:
        p = px[c]
        for w in ventanas:
            vol = p.diff().rolling(w).std().dropna()
            niv = p.rolling(w).mean().reindex(vol.index)
            m = sm.OLS(np.log(vol), sm.add_constant(np.log(niv))).fit(
                cov_type="HAC", cov_kwds={"maxlags": w})       # ventanas solapadas
            lo, hi = m.conf_int().iloc[1]
            filas.append({"serie": c, "ventana": w, "k": m.params.iloc[1], "IC95_inf": lo, "IC95_sup": hi})
    return pd.DataFrame(filas).round(3)


# =====================================================================================
# 02 · Retornos (una serie a la vez)
# =====================================================================================
def estadisticos(serie, nombre=None, lags=10):
    """Estadísticos descriptivos y pruebas para una serie ya transformada (retornos o diferencias)."""
    r = serie.dropna()
    nombre = nombre or serie.name

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")   # KPSS lanza InterpolationWarning en los extremos
        adf_stat, adf_p = adfuller(r, autolag="AIC")[:2]
        kpss_stat, kpss_p = kpss(r, regression="c", nlags="auto")[:2]

    jb_stat, jb_p = stats.jarque_bera(r)
    gl_t, loc_t, esc_t = stats.t.fit(r)
    lb_r = acorr_ljungbox(r, lags=[lags])
    lb_r2 = acorr_ljungbox(r ** 2, lags=[lags])
    lm_stat, lm_p, _, _ = het_arch(r, nlags=lags)

    res = {
        "n": len(r), "media": r.mean(), "std": r.std(),
        "min": r.min(), "p01": r.quantile(0.01), "p05": r.quantile(0.05), "mediana": r.median(),
        "p95": r.quantile(0.95), "p99": r.quantile(0.99), "max": r.max(),
        "asimetria": stats.skew(r), "curtosis_exceso": stats.kurtosis(r),
        "JB_stat": jb_stat, "JB_p": jb_p, "t_gl": gl_t,
        "ADF_stat": adf_stat, "ADF_p": adf_p, "KPSS_stat": kpss_stat, "KPSS_p": kpss_p,
        f"LB_r_p({lags})": lb_r["lb_pvalue"].iloc[0],
        f"LB_r2_p({lags})": lb_r2["lb_pvalue"].iloc[0],
        f"ARCH_LM_p({lags})": lm_p,
        "autocorr_r_lag1": r.autocorr(1), "autocorr_r2_lag1": (r ** 2).autocorr(1),
        "apalancamiento": r.shift(1).corr(r.abs()),   # corr(r(t-1), |r(t)|)
        "VaR_1%": -r.quantile(0.01), "VaR_5%": -r.quantile(0.05),
        "ES_1%": -r[r <= r.quantile(0.01)].mean(), "ES_5%": -r[r <= r.quantile(0.05)].mean(),
    }
    return pd.Series(res, name=nombre)


def tabla_estadisticos(ret, lags=10):
    return pd.concat([estadisticos(ret[c], c, lags) for c in ret.columns], axis=1)


def outliers_z(r, k=None):
    """Observaciones con |z| > k, con media y desviación de la propia serie que se pase (train)."""
    k = cfg.umbral_outlier_z if k is None else k
    r = r.dropna()
    return r[(r - r.mean()).abs() > k * r.std()]


def graficar_serie(r, c, k=None):
    """Serie transformada con los puntos |z| > k marcados."""
    k = cfg.umbral_outlier_z if k is None else k
    r = r.dropna(); out = outliers_z(r, k)
    fig, ax = plt.subplots(figsize=(13, 3.2))
    ax.plot(r, lw=0.5)
    ax.axhline(r.mean(), color="k", ls="--", lw=0.7)
    ax.scatter(out.index, out, color="red", s=16, zorder=3, label=f"|z|>{k}: {len(out)}")
    ax.set_title(f"{c} ({unidad(c)}), train")
    ax.grid(alpha=0.3); ax.legend()
    plt.tight_layout(); plt.show()


def graficar_panel(r, c, w1=21, w2=63):
    """Panel 2×3: serie, histograma, QQ-plot vs t, desviación móvil, media móvil, apalancamiento."""
    r = r.dropna()
    mu, sd = r.mean(), r.std()
    gl, loc_t, esc_t = stats.t.fit(r)
    out = outliers_z(r)

    fig, ax = plt.subplots(2, 3, figsize=(16, 8))
    fig.suptitle(f"{c} ({unidad(c)}), train", fontsize=13)

    ax[0, 0].plot(r, lw=0.5)
    ax[0, 0].scatter(out.index, out, color="red", s=14, zorder=3, label=f"|z|>{cfg.umbral_outlier_z}: {len(out)}")
    ax[0, 0].axhline(mu, color="k", ls="--", lw=0.7); ax[0, 0].legend()
    ax[0, 0].set_title("Serie transformada")

    ax[0, 1].hist(r, bins=80, density=True, alpha=0.6)
    xs = np.linspace(r.min(), r.max(), 400)
    ax[0, 1].plot(xs, stats.norm.pdf(xs, mu, sd), "r", label="Normal")
    ax[0, 1].plot(xs, stats.t.pdf(xs, gl, loc_t, esc_t), "g", label=f"t (gl={gl:.1f})")
    ax[0, 1].legend(); ax[0, 1].set_title(f"asim={stats.skew(r):.2f}  curt={stats.kurtosis(r):.2f}")

    sm.qqplot(r, stats.t, distargs=(gl,), loc=loc_t, scale=esc_t, line="45", ax=ax[0, 2], markersize=2)
    ax[0, 2].set_title("QQ-plot vs t")

    ax[1, 0].plot(r.rolling(w1).std(), lw=0.8, label=f"{w1} d")
    ax[1, 0].plot(r.rolling(w2).std(), lw=1.2, label=f"{w2} d")
    ax[1, 0].axhline(sd, color="k", ls=":", lw=0.8); ax[1, 0].legend()
    ax[1, 0].set_title("Desviación móvil")

    ax[1, 1].plot(r.rolling(w2).mean(), lw=0.9)
    ax[1, 1].axhline(0, color="k", ls=":", lw=0.8)
    ax[1, 1].set_title(f"Media móvil {w2} d")

    ax[1, 2].scatter(r.shift(1), r.abs(), s=3, alpha=0.3)
    ax[1, 2].set_xlabel("r(t-1)"); ax[1, 2].set_ylabel("|r(t)|")
    ax[1, 2].set_title(f"Apalancamiento (corr={r.shift(1).corr(r.abs()):.2f})")
    plt.tight_layout(); plt.show()


def graficar_acf(ret, lags=40):
    """ACF de r, ACF de r² y PACF de r para todas las series."""
    fig, axes = plt.subplots(len(ret.columns), 3, figsize=(13, 2.4 * len(ret.columns)))
    for i, c in enumerate(ret.columns):
        r = ret[c].dropna()
        sm.graphics.tsa.plot_acf(r, lags=lags, ax=axes[i, 0], zero=False, title=f"{c}: ACF de r")
        sm.graphics.tsa.plot_acf(r ** 2, lags=lags, ax=axes[i, 1], zero=False, title=f"{c}: ACF de r²")
        sm.graphics.tsa.plot_pacf(r, lags=lags, ax=axes[i, 2], zero=False, method="ywm", title=f"{c}: PACF de r")
    plt.tight_layout(); plt.show()


def graficar_estacionalidad(r, c):
    """Media por mes y por día de la semana (relevante sobre todo para gas natural)."""
    r = r.dropna()
    fig, ax = plt.subplots(1, 2, figsize=(12, 3.2))
    r.groupby(r.index.month).mean().plot.bar(ax=ax[0], title=f"{c}: media por mes")
    r.groupby(r.index.dayofweek).mean().plot.bar(ax=ax[1], title=f"{c}: media por día (0 = lunes)")
    plt.tight_layout(); plt.show()


def piso_referencia(r, n_splits=4):
    """
    Piso de referencia por pliegue (ventana expansiva, mismos cortes que K_fold del baseline).
    Cualquier modelo debe compararse contra RMSE_cero y RMSE_media_train.
    """
    r = r.dropna(); n = len(r); filas = []
    for i in range(1, n_splits + 1):
        s = (i * n) // (n_splits + 1) + n % (n_splits + 1)
        v = n // (n_splits + 1)
        tr_, va_ = r.iloc[:s], r.iloc[s:s + v]
        filas.append({
            "pliegue": i, "val_ini": va_.index[0].date(), "val_fin": va_.index[-1].date(),
            "std_val": va_.std(),
            "RMSE_cero": np.sqrt((va_ ** 2).mean()),
            "RMSE_media_train": np.sqrt(((va_ - tr_.mean()) ** 2).mean()),
        })
    return pd.DataFrame(filas).round(5)


def escalamiento_horizonte(r, horizontes=(1, 5, 10, 21, 42, 63)):
    """Retornos acumulados a H días (no solapados): desviación, razón de varianzas y curtosis."""
    r = r.dropna(); filas = []
    for H in horizontes:
        rH = r.rolling(H).sum().dropna().iloc[::H]
        filas.append({"H": H, "n": len(rH), "std": rH.std(),
                      "VR": rH.var() / (H * r.var()),          # ≈ 1: retornos casi independientes
                      "curtosis_exc": stats.kurtosis(rH)})
    return pd.DataFrame(filas).round(4)


def comparar_train_val(tr, va):
    """SOLO referencia: ¿se parece la validación al train? No cambia ninguna decisión."""
    filas = []
    for c in tr.columns:
        a, b = tr[c].dropna(), va[c].dropna()
        ks = stats.ks_2samp(a, b)
        filas.append({"serie": c, "std_train": a.std(), "std_val": b.std(),
                      "razon_std": b.std() / a.std(), "KS_stat": ks.statistic, "KS_p": ks.pvalue})
    return pd.DataFrame(filas).round(4)


# =====================================================================================
# 03 · Exógenas
# =====================================================================================
def correlaciones_con_target(ret, target, exog):
    """Pearson y Spearman de cada exógena con el retorno del target (misma fecha)."""
    t = pd.DataFrame({
        "Pearson": ret[exog + [target]].corr(method="pearson")[target].drop(target),
        "Spearman": ret[exog + [target]].corr(method="spearman")[target].drop(target),
    })
    return t.sort_values("Pearson", key=lambda x: x.abs(), ascending=False)


def correlaciones_volatilidad(ret, target, exog):
    """Spearman de cada exógena (con signo y en valor absoluto) con |target|."""
    filas = []
    for c in exog:
        filas.append({
            "exog": c,
            "signo_vs_|WTI|": ret[c].corr(ret[target].abs(), method="spearman"),
            "|x|_vs_|WTI|": ret[c].abs().corr(ret[target].abs(), method="spearman"),
        })
    return pd.DataFrame(filas).set_index("exog").round(4)


def correlacion_rezagada(ret, target, exog, max_lag=10):
    """corr(exógena_{t-k}, target_t) para k = 0..max_lag (k = 0 es contemporánea)."""
    return pd.DataFrame(
        {c: [ret[c].shift(k).corr(ret[target]) for k in range(max_lag + 1)] for c in exog},
        index=pd.RangeIndex(max_lag + 1, name="rezago"))


def graficar_correlaciones(ret, target, exog, max_lag=10):
    corr = ret.corr()
    fig, ax = plt.subplots(1, 3, figsize=(20, 5))
    im = ax[0].imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax[0].set_xticks(range(len(corr))); ax[0].set_xticklabels(corr.columns, rotation=45)
    ax[0].set_yticks(range(len(corr))); ax[0].set_yticklabels(corr.columns)
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax[0].text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
    plt.colorbar(im, ax=ax[0]); ax[0].set_title("Correlación contemporánea (retornos, train)")

    lag = correlacion_rezagada(ret, target, exog, max_lag)
    lag.plot(ax=ax[1], marker="o")
    banda = 1.96 / np.sqrt(len(ret))
    ax[1].axhline(banda, color="r", ls="--", lw=0.8); ax[1].axhline(-banda, color="r", ls="--", lw=0.8)
    ax[1].axhline(0, color="k", lw=0.8); ax[1].set_xticks(range(max_lag + 1)); ax[1].grid(alpha=0.3)
    ax[1].set_title("exógena(t-k) vs WTI(t)  (rojo: ±1.96/√n)")

    for c in exog:
        ret[target].rolling(126).corr(ret[c]).plot(ax=ax[2], lw=0.9, label=c)
    ax[2].axhline(0, color="k", ls=":"); ax[2].legend(ncol=3); ax[2].set_title("Correlación móvil 126 d con el WTI")
    plt.tight_layout(); plt.show()


def granger_tabla(ret, target, exog, maxlag=5, volatilidad=False):
    """
    p-valores de Granger (F, 'ssr_ftest'): ¿los rezagos de la exógena ayudan a predecir el target?
    volatilidad=False: exógena con signo -> target.  volatilidad=True: {signo, |x|} -> |target|.
    """
    filas = []
    for c in exog:
        variantes = {"signo": ret[c], "abs": ret[c].abs()} if volatilidad else {"signo": ret[c]}
        y = ret[target].abs() if volatilidad else ret[target]
        for nombre, x in variantes.items():
            d = pd.concat([y.rename("y"), x.rename("x")], axis=1).dropna()
            res = grangercausalitytests(d, maxlag=maxlag)
            for k in range(1, maxlag + 1):
                filas.append({"exog": c, "pred": nombre, "rezago": k, "p": res[k][0]["ssr_ftest"][1]})
    idx = ["exog", "pred"] if volatilidad else ["exog"]
    tab = pd.DataFrame(filas).pivot_table(index=idx, columns="rezago", values="p")
    n_pruebas = len(filas)
    return tab, 0.05 / n_pruebas       # (tabla, umbral de Bonferroni)


def vif_exogenas(ret, exog):
    """Colinealidad entre exógenas: > 5 preocupa, > 10 es grave."""
    X = sm.add_constant(ret[exog].dropna())
    return pd.Series([variance_inflation_factor(X.values, i) for i in range(1, X.shape[1])],
                     index=exog, name="VIF").round(2)


def _spearman_circular(x, y, nperm, rng):
    """Spearman con p-valor por desplazamiento circular de x (conserva autocorrelación y volatilidad)."""
    rx, ry = stats.rankdata(x), stats.rankdata(y)
    rx, ry = (rx - rx.mean()) / rx.std(), (ry - ry.mean()) / ry.std()
    obs = np.mean(rx * ry)
    n = len(rx)
    desplaz = rng.integers(21, n - 21, nperm)
    nulo = np.array([np.mean(np.roll(rx, s) * ry) for s in desplaz])
    return obs, (1 + np.sum(np.abs(nulo) >= abs(obs))) / (1 + nperm)


def dependencia_circular(ret, target, exog, lags=range(1, 6), nperm=2000, seed=0):
    """Spearman rezagado (exógena en t-k, target en t) con p-valor circular y corrección FDR."""
    rng = np.random.default_rng(seed)
    filas = []
    for c in exog:
        for relacion, (X, Y) in {
            "signo→retorno": (ret[c], ret[target]),
            "|x|→|WTI|":     (ret[c].abs(), ret[target].abs()),
            "signo→|WTI|":   (ret[c], ret[target].abs()),
        }.items():
            for k in lags:
                rho, p = _spearman_circular(X.values[:-k], Y.values[k:], nperm, rng)
                filas.append({"exog": c, "relacion": relacion, "rezago": k, "rho": rho, "p": p})
    out = pd.DataFrame(filas)
    out["q_FDR"] = multipletests(out["p"], method="fdr_bh")[1]
    return out


def lift_colas(ret, target, exog, q=0.90, nperm=2000, seed=0):
    """P(día extremo del target | exógena extrema ayer) / P(día extremo del target). 1 = sin relación."""
    rng = np.random.default_rng(seed)
    yext = (ret[target].abs() > ret[target].abs().quantile(q)).astype(float)
    filas = []
    for c in exog:
        xext = (ret[c].abs() > ret[c].abs().quantile(q)).astype(float).shift(1)
        d = pd.concat([xext, yext], axis=1).dropna()
        a, b = d.iloc[:, 0].values, d.iloc[:, 1].values
        lift = b[a == 1].mean() / b.mean()
        nulo = np.array([b[np.roll(a, s) == 1].mean() / b.mean() for s in rng.integers(21, len(a) - 21, nperm)])
        filas.append({"exog": c, "lift": lift, "p": (1 + np.sum(nulo >= lift)) / (1 + nperm)})
    out = pd.DataFrame(filas)
    out["q_FDR"] = multipletests(out["p"], method="fdr_bh")[1]
    return out.round(4)



def graficar_series_transformadas(ret, w=63):
    """Todas las series transformadas apiladas (mismo eje de fechas), con la desviación móvil w d en la banda ±."""
    cols = list(ret.columns)
    fig, axes = plt.subplots(len(cols), 1, figsize=(13, 2.2 * len(cols)), sharex=True)
    for ax, c in zip(np.atleast_1d(axes), cols):
        r = ret[c]
        ax.plot(r, lw=0.5)
        ax.axhline(r.mean(), color="k", ls="--", lw=0.7)
        sd = r.rolling(w).std()
        ax.plot(sd, color="red", lw=0.8, label=f"±σ móvil {w} d")
        ax.plot(-sd, color="red", lw=0.8)
        ax.set_ylabel(f"{c}\n({unidad(c)})", fontsize=8); ax.grid(alpha=0.3)
        ax.legend(loc="upper right", fontsize=7)
    np.atleast_1d(axes)[0].set_title("Series transformadas, train")
    plt.tight_layout(); plt.show()


def pruebas_dependencia(ret, lags=(5, 10, 20), alpha=0.05):
    """
    Pruebas formales que acompañan a la ACF / PACF, para cada serie transformada (usar con train).

      Ljung-Box(r)    H0: sin autocorrelación conjunta en r hasta el rezago m.
      Ljung-Box(r²)   H0: sin autocorrelación conjunta en r² (dependencia en la magnitud de los movimientos).
      ARCH-LM         H0: sin efecto ARCH (la varianza no depende de los errores² pasados).
                      Se aplica a e = r - media (residuo de una ecuación de media constante).
      Jarque-Bera     H0: asimetría y curtosis de una normal.
      ADF             H0: raíz unitaria (no estacionaria).      Rechazar H0 = evidencia de estacionariedad.
      KPSS            H0: estacionaria alrededor de una media.  Rechazar H0 = evidencia contra estacionariedad.

    Devuelve una tabla larga: serie, prueba, rezagos, estadístico, p_valor, H0, rechaza_H0, lectura.
    ADF y KPSS y JB no dependen del rezago m (rezagos = '-'): ADF elige sus rezagos por AIC y KPSS usa nlags='auto'.
    El p-valor de KPSS está acotado por statsmodels entre 0.01 y 0.10.
    """
    filas = []

    def agregar(c, prueba, m, est, p, h0, lectura_si_rechaza, lectura_si_no):
        rech = bool(p < alpha)
        filas.append({"serie": c, "prueba": prueba, "rezagos": m, "estadistico": est, "p_valor": p,
                      "H0": h0, "rechaza_H0": rech,
                      "lectura": lectura_si_rechaza if rech else lectura_si_no})

    for c in ret.columns:
        r = ret[c].dropna()
        e = r - r.mean()
        lb_r = acorr_ljungbox(r, lags=list(lags))
        lb_r2 = acorr_ljungbox(r ** 2, lags=list(lags))
        for m in lags:
            agregar(c, "Ljung-Box r", m, lb_r.loc[m, "lb_stat"], lb_r.loc[m, "lb_pvalue"],
                    "sin autocorrelación en r", "hay autocorrelación en r", "sin autocorrelación en r")
            agregar(c, "Ljung-Box r²", m, lb_r2.loc[m, "lb_stat"], lb_r2.loc[m, "lb_pvalue"],
                    "sin autocorrelación en r²", "la magnitud de los movimientos es predecible (volatilidad agrupada)",
                    "sin dependencia en la magnitud")
            lm, p_lm, _, _ = het_arch(e, nlags=m)
            agregar(c, "ARCH-LM", m, lm, p_lm,
                    "sin efecto ARCH", "efecto ARCH: la varianza depende del pasado", "sin efecto ARCH")

        jb, p_jb = stats.jarque_bera(r)
        agregar(c, "Jarque-Bera", "-", jb, p_jb, "distribución normal",
                "se aparta de la normal (asimetría y/o colas)", "compatible con la normal")

        adf_stat, adf_p = adfuller(r, autolag="AIC")[:2]
        agregar(c, "ADF", "-", adf_stat, adf_p, "raíz unitaria",
                "estacionaria (rechaza la raíz unitaria)", "no se rechaza la raíz unitaria")

        kpss_stat, kpss_p = kpss(r, regression="c", nlags="auto")[:2]
        agregar(c, "KPSS", "-", kpss_stat, kpss_p, "estacionaria",
                "evidencia contra la estacionariedad", "no se rechaza la estacionariedad")

    out = pd.DataFrame(filas)
    out["estadistico"] = out["estadistico"].astype(float).round(4)
    out["p_valor"] = out["p_valor"].astype(float)
    return out


def resumen_pruebas(tabla, rezago_ref=10):
    """
    p-valores en formato ancho (filas = serie, columnas = prueba) para un rezago de referencia.
    Las pruebas sin rezago (JB, ADF, KPSS) se muestran siempre.
    """
    t = tabla[(tabla["rezagos"] == rezago_ref) | (tabla["rezagos"] == "-")]
    orden = ["Ljung-Box r", "Ljung-Box r²", "ARCH-LM", "Jarque-Bera", "ADF", "KPSS"]
    ancho = t.pivot(index="serie", columns="prueba", values="p_valor")[orden]
    return ancho.reindex(tabla["serie"].unique())
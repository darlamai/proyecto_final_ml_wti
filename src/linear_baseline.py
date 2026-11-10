"""
Modelo base lineal (el mismo del cuaderno Linear_Model_Baseline.ipynb) para series de retornos.

Regresión lineal directa multisalida: dados los últimos K valores de la serie, predice los H siguientes
con la ecuación normal  W = (XᵀX)⁻¹ Xᵀ Y.  K se elige con validación cruzada de ventana expansiva
(K_fold) usando solo train.

Cambios respecto al cuaderno original (todos para que funcione sobre retornos y sin fugas):
  * RMSE_CV usaba las listas globales `data_train_lst` / `data_validation_lst`; ahora usa las que
    construye con el conjunto de entrenamiento que se le pasa.
  * K_fold usaba el `n_splits` global; ahora usa su argumento.
  * calculate_rmse ignora las ventanas que contienen NaN (el retorno del WTI del 20 y 21/04/2020 no existe).
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ------------------------------------------------------------------------------------------
# Validación cruzada de ventana expansiva (igual que el cuaderno)
# ------------------------------------------------------------------------------------------
def K_fold(training_dataset, n_splits):
    n_D = len(training_dataset)
    D_train_lst, D_validation_lst = [], []
    for i in range(1, n_splits + 1):
        s_train_i = i * n_D // (n_splits + 1) + n_D % (n_splits + 1)
        s_validation_i = n_D // (n_splits + 1)
        D_train_lst.append(training_dataset.iloc[0: s_train_i])
        D_validation_lst.append(training_dataset.iloc[s_train_i: s_train_i + s_validation_i])
    return D_train_lst, D_validation_lst


def create_sequences(data, steps_back, steps_ahead):
    """Ventanas deslizantes: X = K valores, Y = los H siguientes. Devuelve (N, K) y (N, H)."""
    X, Y = [], []
    data = np.array(data)
    for i in range(len(data) - steps_back - steps_ahead + 1):
        X.append(data[i: (i + steps_back)])
        Y.append(data[(i + steps_back): (i + steps_back + steps_ahead)])
    return np.array(X), np.array(Y)


# ------------------------------------------------------------------------------------------
# Modelo
# ------------------------------------------------------------------------------------------
def normal_equation(X, Y):
    """Solución cerrada de la regresión lineal multientrada y multisalida. W tiene forma (D + 1, K)."""
    X_design = np.hstack([np.ones((X.shape[0], 1)), X])
    return np.linalg.pinv(X_design.T @ X_design) @ X_design.T @ Y


def predict(X, W):
    X_design = np.hstack([np.ones((X.shape[0], 1)), X])
    return X_design @ W


def ventanas_validas(Y, Y_hat):
    """True en las filas sin NaN ni en Y ni en Y_hat."""
    return ~(np.isnan(np.asarray(Y)).any(axis=1) | np.isnan(np.asarray(Y_hat)).any(axis=1))


def _referencia(Y, ok, valor):
    """Pronóstico constante `valor`, con NaN en las ventanas que el modelo no puede predecir:
    así el modelo y las referencias se comparan sobre exactamente las mismas ventanas."""
    R = np.full(np.shape(Y), float(valor))
    R[~ok] = np.nan
    return R


def calculate_rmse(Y, Y_hat):
    """RMSE de Frobenius, sobre las ventanas sin NaN."""
    Y, Y_hat = np.array(Y), np.array(Y_hat)
    ok = ventanas_validas(Y, Y_hat)
    Y, Y_hat = Y[ok], Y_hat[ok]
    N = Y.shape[0] * Y.shape[1]
    return np.linalg.norm(Y - Y_hat, ord="fro") / N ** 0.5


# ------------------------------------------------------------------------------------------
# Selección de K con validación cruzada (solo con train)
# ------------------------------------------------------------------------------------------
def RMSE_CV(steps_back, steps_ahead, training_dataset, n_splits):
    D_train_lst, D_validation_lst = K_fold(training_dataset, n_splits)
    n_splits = len(D_train_lst)
    rmse_tr, rmse_va = 0.0, 0.0
    for i in range(n_splits):
        X_tr, Y_tr = create_sequences(D_train_lst[i], steps_back, steps_ahead)
        X_va, Y_va = create_sequences(D_validation_lst[i], steps_back, steps_ahead)
        W = normal_equation(X_tr, Y_tr)
        rmse_tr += calculate_rmse(Y_tr, predict(X_tr, W))
        rmse_va += calculate_rmse(Y_va, predict(X_va, W))
    return rmse_tr / n_splits, rmse_va / n_splits


def RMSE_CV_referencia(steps_ahead, training_dataset, n_splits, steps_back=1):
    """RMSE medio del pronóstico 'cero' en los mismos pliegues de validación (piso de referencia)."""
    _, D_validation_lst = K_fold(training_dataset, n_splits)
    total = 0.0
    for D_va in D_validation_lst:
        _, Y_va = create_sequences(D_va, steps_back, steps_ahead)
        total += calculate_rmse(Y_va, np.zeros_like(Y_va))
    return total / len(D_validation_lst)


def plot_RMSE_steps_ahead(k_values, RMSE_training_lst, RMSE_validation_lst, Curve):
    """Curva de RMSE contra K ('Training' o 'Validation') con el mejor K marcado."""
    if Curve not in ("Training", "Validation"):
        return
    lst = RMSE_training_lst if Curve == "Training" else RMSE_validation_lst
    color = "green" if Curve == "Training" else "purple"
    etiqueta = "Entrenamiento" if Curve == "Training" else "Validación"
    k_best = list(k_values)[int(np.argmin(lst))]
    plt.figure(figsize=(10, 6))
    plt.plot(list(k_values), lst, marker="o", label=f"RMSE de {etiqueta}")
    plt.scatter(k_best, min(lst), color=color, s=100, zorder=5, label=f"Mejor K {etiqueta} ({k_best})")
    plt.annotate(f"K={k_best}", (k_best, min(lst)), textcoords="offset points", xytext=(0, 10), ha="center", color=color)
    plt.xlabel("Número de pasos atrás (K)"); plt.ylabel("RMSE")
    plt.title("RMSE con diferente número de pasos atrás")
    plt.legend(); plt.grid(True); plt.show()


def RMSE_CV_search_K(k_values, steps_ahead, training_dataset, n_splits, plot=None, verbose=True):
    """Devuelve (mejor K según RMSE de validación, RMSE_train_por_K, RMSE_val_por_K)."""
    k_values = list(k_values)
    tr, va = [], []
    for k in k_values:
        a, b = RMSE_CV(k, steps_ahead, training_dataset, n_splits)
        tr.append(a); va.append(b)
    best = k_values[int(np.argmin(va))]
    if verbose:
        print(f"El K con menor RMSE de validación es {best} (RMSE: {min(va):.6f})")
    if plot:
        plot_RMSE_steps_ahead(k_values, tr, va, Curve=plot)
    return best, tr, va


# ------------------------------------------------------------------------------------------
# Evaluación fuera de la muestra (validación 2019 o test)
# ------------------------------------------------------------------------------------------
def con_prefijo(anterior, segmento, steps_back):
    """
    Une los últimos K datos del tramo ANTERIOR con el segmento a evaluar (en el cuaderno: data_combined).
    Así la primera predicción del segmento ya tiene sus K entradas, y toda salida cae dentro del segmento.
    Para val el tramo anterior es train; para test es val.
    """
    return pd.concat([anterior.iloc[-steps_back:], segmento])


def evaluar_segmento(anterior, segmento, W, steps_back, steps_ahead, mu_train=0.0):
    """
    Predice todas las ventanas cuya salida cae en `segmento`. Devuelve un diccionario con X, Y, Y_hat,
    la fecha de inicio de cada ventana y el RMSE del modelo, del pronóstico 'cero' y de la media de train
    (las tres sobre las mismas ventanas sin NaN).
    """
    comb = con_prefijo(anterior, segmento, steps_back)
    X, Y = create_sequences(comb, steps_back, steps_ahead)
    Y_hat = predict(X, W)
    ok = ventanas_validas(Y, Y_hat)
    fechas = segmento.index[: len(Y)]
    return {
        "X": X, "Y": Y, "Y_hat": Y_hat, "fechas": fechas, "validas": ok,
        "n_ventanas": int(ok.sum()), "n_descartadas": int((~ok).sum()),
        "RMSE_modelo": calculate_rmse(Y, Y_hat),
        "RMSE_cero": calculate_rmse(Y, _referencia(Y, ok, 0.0)),
        "RMSE_media_train": calculate_rmse(Y, _referencia(Y, ok, mu_train)),
    }


def rmse_por_ventana(res, ventanas, mu_train=0.0):
    """RMSE en cada sub-ventana (ej. config.VENTANAS_TEST); una ventana de predicción pertenece a la
    sub-ventana donde empieza su salida."""
    filas = []
    for nombre, (a, b) in ventanas.items():
        m = (res["fechas"] >= pd.Timestamp(a)) & (res["fechas"] <= pd.Timestamp(b))
        Y, Yh = res["Y"][m], res["Y_hat"][m]
        ok = ventanas_validas(Y, Yh)
        filas.append({
            "ventana": nombre, "ventanas_validas": int(ok.sum()), "descartadas_por_NaN": int((~ok).sum()),
            "RMSE_modelo": calculate_rmse(Y, Yh) if ok.any() else np.nan,
            "RMSE_cero": calculate_rmse(Y, _referencia(Y, ok, 0.0)) if ok.any() else np.nan,
            "RMSE_media_train": calculate_rmse(Y, _referencia(Y, ok, mu_train)) if ok.any() else np.nan,
        })
    return pd.DataFrame(filas).set_index("ventana")


def reconstruir_vector(Y_hat, index):
    """
    Una predicción por fecha, como en el cuaderno: bloques de H días sin solapes (la ventana 0, H, 2H, ...)
    y los días restantes tomados de la última ventana.
    """
    N, H = Y_hat.shape
    T = N + H - 1
    starts = np.arange(0, N, H)
    bloques = Y_hat[starts].ravel()
    r = T - bloques.size
    vec = np.concatenate([bloques, Y_hat[-1, H - r:]])
    return pd.Series(vec, index=index[:T], name="Predicho")


def pronostico_futuro(segmento, W, steps_back):
    """H valores después del último dato, a partir de los últimos K (el 'futuro' del cuaderno)."""
    X_future = np.array(segmento.iloc[-steps_back:]).reshape(1, -1)
    Y_future = predict(X_future, W)
    fechas = pd.bdate_range(start=segmento.index[-1] + pd.Timedelta(days=1), periods=Y_future.shape[1])
    return pd.Series(Y_future[0], index=fechas, name="Predicho")


# ------------------------------------------------------------------------------------------
# Gráficos train / validación / test / predicción
# ------------------------------------------------------------------------------------------
def r2_fuera_de_muestra(rmse_modelo, rmse_referencia):
    """R²_OS = 1 - MSE_modelo / MSE_referencia. > 0: el modelo supera a la referencia; < 0: queda peor."""
    return 1.0 - (rmse_modelo / rmse_referencia) ** 2


def graficar_train_test_prediccion(y_train, y_val, y_test, pred_val, pred_test, titulo=""):
    """
    Panel superior: retornos observados de train, validación y test, con la predicción encima.
    Panel inferior: solo la predicción, con su propia escala (los retornos diarios son mucho mayores
    que la predicción y esta no se vería en el panel superior).
    """
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(15, 8), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    a1.plot(y_train, color="0.45", lw=0.5, label="Train (observado)")
    a1.plot(y_val, color="tab:orange", lw=0.5, label="Validación (observado)")
    a1.plot(y_test, color="tab:blue", lw=0.5, label="Test (observado)")
    a1.plot(pred_val, color="red", lw=1.0)
    a1.plot(pred_test, color="red", lw=1.0, label="Predicción")
    a1.set_ylabel("Log-retorno diario"); a1.set_title(f"Retorno del WTI: train, test y predicción {titulo}")
    a1.legend(loc="upper left", ncol=4); a1.grid(alpha=0.3)

    a2.plot(pred_val, color="red", lw=0.8)
    a2.plot(pred_test, color="red", lw=0.8)
    a2.axhline(0, color="k", lw=0.6, ls=":")
    a2.set_ylabel("Predicción"); a2.set_xlabel("Tiempo"); a2.grid(alpha=0.3)
    a2.set_title("Solo la predicción (escala propia)")
    for ax in (a1, a2):
        for corte in (y_train.index[-1], y_val.index[-1]):
            ax.axvline(corte, color="k", ls="--", lw=0.7)
    plt.tight_layout(); plt.show()


def graficar_precio_prediccion(px_train, px_val, px_test, pred_val, pred_test, H=1):
    """
    Precio del WTI (train, validación, test) y precio previsto un día adelante:
        P_prev(t) = P_observado(t-1) · exp(retorno previsto(t)).
    Solo vale para H = 1: con H > 1 habría que encadenar previsiones y re-anclar el precio, y el gráfico
    de retornos ya muestra la predicción sin ambigüedad. Aviso: como el precio previsto parte del precio
    observado de ayer, queda pegado a la curva real aunque el retorno previsto sea casi cero.
    """
    if H != 1:
        raise ValueError("graficar_precio_prediccion solo está definido para H = 1.")
    px = pd.concat([px_train, px_val, px_test])
    pred = pd.concat([pred_val, pred_test])
    p_hat = px.shift(1).reindex(pred.index) * np.exp(pred)

    fig, ax = plt.subplots(figsize=(15, 5))
    ax.plot(px_train, color="0.45", lw=0.8, label="Train (observado)")
    ax.plot(px_val, color="tab:orange", lw=0.8, label="Validación (observado)")
    ax.plot(px_test, color="tab:blue", lw=0.8, label="Test (observado)")
    ax.plot(p_hat, color="red", lw=0.7, alpha=0.8, label="predicción a 1 día")
    ax.axhline(0, color="k", lw=0.5, ls=":")
    ax.set_xlabel("Tiempo"); ax.set_ylabel("Precio (USD/barril)")
    ax.set_title("Precio del WTI: train, test y predicción a 1 día (a partir del retorno previsto)")
    ax.legend(loc="upper left", ncol=4); ax.grid(alpha=0.3)
    plt.tight_layout(); plt.show()
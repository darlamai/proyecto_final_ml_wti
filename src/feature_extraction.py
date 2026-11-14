def construir_features(base_ret, col_r="WTI", col_park="WTI_park",
                       n_lags=10, ventanas=(5, 21, 63)):
    """
    Extracción de características por rezagos para pronosticar la varianza diaria.

    Convención: la fila del día τ tiene como etiqueta el Parkinson del día τ
    y como variables explicativas solo información disponible hasta τ-1.

    Parameters
    ----------
    base_ret : pd.DataFrame
        Retornos logarítmicos en % (col_r) y estimador de Parkinson en %² (col_park),
        indexados por fecha.
    n_lags : int
        Número de rezagos individuales (días τ-1, ..., τ-n_lags).
    ventanas : tuple
        Ventanas (en días) para la volatilidad móvil y el Parkinson promedio.

    Returns
    -------
    X : pd.DataFrame  variables candidatas
    y : pd.Series     etiqueta (Parkinson del día τ)
    """
    r = base_ret[col_r]
    park = base_ret[col_park]
    X = pd.DataFrame(index=base_ret.index)

    # 1. Rezagos individuales
    for k in range(1, n_lags + 1):
        X[f"r2_lag{k}"]   = (r**2).shift(k)      # retorno al cuadrado
        X[f"absr_lag{k}"] = r.abs().shift(k)     # retorno en valor absoluto
        X[f"park_lag{k}"] = park.shift(k)        # Parkinson de días anteriores

    # 2. Promedios móviles hasta τ-1 (semana, mes, trimestre)
    for w in ventanas:
        X[f"vol_r_{w}"]      = r.rolling(w).std().shift(1)
        X[f"park_media_{w}"] = park.rolling(w).mean().shift(1)

    # 3. Retorno de ayer con signo (efecto apalancamiento)
    X["r_lag1"] = r.shift(1)

    # Etiqueta y limpieza de los primeros días sin historia suficiente
    y = park.rename("y")
    datos = pd.concat([X, y], axis=1).dropna()
    return datos.drop(columns="y"), datos["y"]
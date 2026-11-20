# =============================================================================
# src/feature_extraction.py
# Extracción de características (feature extraction) para el Random Forest.
#
# Contenido:
#   - construir_features: construye las variables predictoras por rezagos a partir de
#     los retornos logarítmicos y del estimador de Parkinson, y la etiqueta (Parkinson
#     del día τ). Cada fila τ usa solo información disponible hasta τ-1, por lo que no
#     hay fuga de información. Con n_lags = 20 y ventanas de 5, 21 y 63 días se obtienen
#     67 variables candidatas:
#       · 20 rezagos de r², 20 de |r| y 20 del estimador de Parkinson
#       · volatilidad móvil (desviación estándar de r) y promedio móvil de Parkinson
#         en ventanas de 5, 21 y 63 días
#       · el retorno del día anterior con signo (efecto apalancamiento)
# Uso de asistentes de IA (requisito de la rúbrica):
#   Partes de este código se generaron o adaptaron con ayuda de un asistente de IA: Claude
# =============================================================================


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src import feature_extraction as fe
from src import config as cfg


# Prompt (Claude, 14-nov-2026): "ayudame con el feature selection con Lags , yo tengo mi data
#   asi , cual serian los rezagos , YA CONSTRUI PARKINSON, ENTRARIA EN FEATURE SELECTION?"
# Prompt (Claude, 14-nov-2026): "ayudame con el còdigo para el feature extraction"
# Uso: extracción clásica por rezagos; en el proyecto se llama con n_lags = 20.
# Referencias: [1], [2], [3], [4], [5], [6]



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
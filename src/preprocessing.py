"""
Módulo de preprocesamiento de datos financieros.

Implementa un flujo reproducible de preparación de datos, independiente
de los notebooks de análisis exploratorio y modelización.

Ejecución
---------
Desde el directorio raíz del proyecto:

    python -m src.preprocessing

Descripción
-----------
El módulo implementa las siguientes etapas del preprocesamiento:

1. Carga de los datos financieros crudos.
2. Eliminación de observaciones con valores nulos.
3. Transformación de las series de precios a retornos logarítmicos
   o primeras diferencias, según las transformaciones configuradas.
4. Partición temporal de los datos en conjuntos de entrenamiento
   (train), validación (val) y prueba (test).
5. Verificación de la integridad de las particiones.
6. Almacenamiento de los conjuntos procesados en formato CSV.

Archivos generados
------------------
* precios_train.csv, precios_val.csv, precios_test.csv
* retornos_train.csv, retornos_val.csv, retornos_test.csv
* filas_eliminadas_por_nulos.csv

Dependencias internas
---------------------
* src.config: configuración de rutas, particiones y transformaciones.
* src.load_clean: funciones de carga, limpieza y partición de datos.
"""


import numpy as np
import pandas as pd
from . import config as cfg
from . import load_clean 


def build_returns(precios, transf=None):
    """ 
    Transformación de las variables según config.TRANSF: WTI-> logret,
    VIX->logret, DXY->logret, GOLD->logret, SP500->logret, NG->logret, TNX->diff,  
    Los precios <= 0 quedan en NaN en 'logret' (no se inventan valores).

    Parameters
    ----------
    precios : pd.DataFrame
        DataFrame con las series históricas de precios. Su índice representa
        las fechas y sus columnas los activos o variables financieras.

    transf : dict, optional
        Diccionario que especifica la transformación que se aplicará a cada
        variable. Admite "logret" para retornos logarítmicos y "diff" para
        primeras diferencias. Si es None o está vacío, utiliza cfg.transf.

    Returns
    -------
    pd.DataFrame
        DataFrame con las series transformadas, conservando el índice
        temporal y las columnas correspondientes a las variables procesadas.
        Se elimina la primera observación porque no dispone de un valor
        anterior para calcular la transformación.

    """
    transf = transf or cfg.transf
    out = pd.DataFrame(index=precios.index)
    for c, t in transf.items():
        s = precios[c]
        if t == "logret":
            out[c] = np.log(s.where(s > 0)).diff()
        elif t == "diff":
            out[c] = s.diff()
        else:
            raise ValueError(f"Transformación desconocida para {c}: {t}")
    return out.iloc[1:]          


def prepare_and_save(ruta_raw=None, destino=None):
    """
    Ejecuta el proceso de preparación de datos financieros y guarda los
    conjuntos procesados en disco.

    El proceso comprende la carga de datos crudos, eliminación de filas
    con valores nulos, cálculo de retornos, partición temporal en conjuntos
    de entrenamiento, validación y prueba, y verificación de las particiones.
    Guarda precios_{train,val,test}.csv, retornos_{train,val,test}.csv y las filas eliminadas.

    Parameters
    ----------
    ruta_raw : str or pathlib.Path, optional
        Ruta del archivo que contiene los datos crudos de precios.
        Si es None, utiliza la ruta predeterminada definida en la función
        de carga de datos.

    destino : str or pathlib.Path, optional
        Directorio donde se almacenarán los archivos procesados.
        Si es None, utiliza cfg.procesados. El directorio se crea
        automáticamente si no existe.

    Returns
    -------
    dict
        Diccionario con el número de observaciones de cada conjunto
        de retornos (train, val y test), además del número de filas
        eliminadas durante la limpieza por presencia de valores nulos.
        Las claves corresponden a los nombres de las particiones y
        a "filas_eliminadas".
    """
    destino = destino or cfg.procesados
    destino.mkdir(parents=True, exist_ok=True)

    base, eliminadas = load_clean.drop_missing_rows(load_clean.load_raw(ruta_raw))
    ret = build_returns(base)

    px_partes, ret_partes = load_clean.split_dataset(base), load_clean.split_dataset(ret)
    load_clean.validate_splits(base, px_partes)
    load_clean.validate_splits(ret, ret_partes)

    for k in cfg.splits:
        px_partes[k].to_csv(destino / f"precios_{k}.csv")
        ret_partes[k].to_csv(destino / f"retornos_{k}.csv")
    eliminadas.to_csv(destino / "filas_eliminadas_por_nulos.csv")

    return {k: len(v) for k, v in ret_partes.items()} | {"filas_eliminadas": len(eliminadas)}


def load_dataset(tipo, split, permitir_test=False, carpeta=None):

    """
    Carga  un conjunto de datos previamente procesado,
    correspondiente al tipo de serie (original o retornos) y partición temporal especificados.
    El acceso al conjunto de prueba (test) está restringido para evitar
    su utilización durante las etapas de entrenamiento y validación.
    Su carga requiere habilitar explícitamente el parámetro permitir_test.

    Parameters
    ----------
    tipo : str
        Tipo de conjunto de datos que se desea cargar.
        Admite "retornos" para las series de retornos o "precios"
        para las series de precios.

    split : str
        Partición temporal que se desea cargar.
        Admite "train" para entrenamiento, "val" para validación
        y "test" para evaluación final.

    permitir_test : bool, optional
        Permite habilitar explícitamente la carga del conjunto de prueba.
        Por defecto es False, por lo que el acceso a test está bloqueado.

    carpeta : str or pathlib.Path, optional
        Directorio donde se encuentran almacenados los archivos procesados.
        Si es None, utiliza el directorio definido en cfg.procesados.

    Returns
    -------
    pd.DataFrame
        DataFrame con los datos correspondientes al tipo y partición
        solicitados. La primera columna se utiliza como índice temporal
        y se convierte a formato datetime.

    """

    if split == "test" and not permitir_test:
        raise PermissionError(
            "El test está reservado para la evaluación final. "
            "Usa permitir_test=True solo en el notebook de evaluación."
        )
    carpeta = carpeta or cfg.procesados
    return pd.read_csv(carpeta / f"{tipo}_{split}.csv", index_col=0, parse_dates=True)


if __name__ == "__main__":
    print(prepare_and_save())
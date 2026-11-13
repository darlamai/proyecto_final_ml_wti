"""
Módulo de carga, limpieza y partición de datos financieros.

Implementa las funciones auxiliares necesarias para la lectura,
el diagnóstico de valores faltantes, la limpieza y la partición
temporal de las series financieras utilizadas en el proyecto.

Este módulo forma parte del pipeline de preprocesamiento y proporciona
las operaciones fundamentales que posteriormente son coordinadas
por el módulo preprocessing.py.

Descripción
-----------
El módulo comprende las siguientes operaciones:

1. Carga de datos financieros crudos desde archivos CSV.
2. Conversión y organización del índice temporal de las observaciones.
3. Selección de las variables financieras definidas en la configuración.
4. Diagnóstico de valores faltantes mediante el cálculo de cantidades
   y porcentajes de valores nulos por variable.
5. Evaluación y eliminación de observaciones incompletas según
   el umbral de valores faltantes establecido.
6. Partición temporal de los datos en conjuntos de entrenamiento
   (train), validación (val) y prueba (test).
7. Validación de la consistencia temporal y la integridad
   de las particiones generadas.

   
Dependencias internas
---------------------
* src.config: configuración de rutas, variables financieras,
  umbrales de limpieza, intervalos de partición y ventanas de evaluación.

Dependencias externas
---------------------
* pandas: lectura, manipulación, limpieza y partición de DataFrames.   

"""

import pandas as pd

from . import config as cfg


def load_raw(ruta=None):
    """
    Carga los datos financieros crudos desde un archivo CSV y prepara
    su estructura temporal para el procesamiento posterior.

    Convierte la columna Date al formato datetime, la establece como
    índice del DataFrame y ordena las observaciones cronológicamente.
    Finalmente, conserva únicamente las columnas especificadas (campo: columns)
    en la configuración del proyecto: target= WTI,
    exogenas= VIX, DXY, GOLD, TNX, SP500,NG

    Parameters
    ----------
    ruta : str or pathlib.Path, optional
        Ruta del archivo CSV que contiene los datos financieros crudos.
        Si es None, utiliza la ruta predeterminada definida en cfg.raw_csv.

    Returns
    -------
    pd.DataFrame
        DataFrame con las series históricas de precios seleccionadas.
        El índice corresponde a las fechas en formato DatetimeIndex,
        ordenadas cronológicamente, y las columnas corresponden a las
        variables financieras definidas en cfg.columns.
    """    
    ruta = ruta or cfg.raw_csv
    # Si la ruta es un directorio, buscar el CSV disponible
    if ruta.is_dir():
        archivos_csv = sorted(
            ruta.glob("*.csv"),
            key=lambda archivo: archivo.stat().st_mtime,
            reverse=True,
        )

        if not archivos_csv:
            raise FileNotFoundError(
                f"No se encontró ningún archivo CSV en {ruta}"
            )

        ruta_seleccionada = None
        for archivo in archivos_csv:
            columnas = pd.read_csv(archivo, nrows=0).columns
            if set(cfg.columns).issubset(columnas):
                ruta_seleccionada = archivo
                break

        if ruta_seleccionada is None:
            raise ValueError(
                f"Ningún CSV en {ruta} contiene las columnas requeridas: "
                f"{cfg.columns}"
            )

        ruta = ruta_seleccionada

    base = pd.read_csv(ruta)
    faltantes = [columna for columna in cfg.columns if columna not in base.columns]
    if faltantes:
        raise ValueError(
            f"El archivo {ruta} no contiene las columnas requeridas: {faltantes}"
        )
    base["Date"] = pd.to_datetime(base["Date"])
    base = base.set_index("Date").sort_index()
    return base[cfg.columns]


def summarize_missing_values(base):
    """
    Calcula un resumen estadístico de los valores faltantes
    presentes en cada columna de un DataFrame.
    Para cada variable, determina la cantidad de observaciones nulas
    y su proporción porcentual respecto al total de filas.
    Los resultados se ordenan de mayor a menor según la cantidad
    de valores faltantes.

    Parameters
    ----------
    base : pd.DataFrame
        DataFrame que contiene las observaciones que se desean analizar.
        Puede incluir variables numéricas, categóricas o temporales.

    Returns
    -------
    pd.DataFrame
        DataFrame con dos columnas:

        - cantidad : int
            Número de valores faltantes por variable.
        - porcentaje : float
            Porcentaje de valores faltantes respecto al total
            de observaciones, redondeado a dos decimales.
    """
    return pd.DataFrame({
        "cantidad": base.isna().sum(),
        "porcentaje": (base.isna().mean() * 100).round(2),
    }).sort_values("cantidad", ascending=False)


def summarize_negative_values(base):
    """
    Calcula un resumen de los valores negativos presentes en cada columna
    numérica de un DataFrame.

    Para cada variable numérica, determina la cantidad de valores menores
    que cero y su porcentaje respecto al total de filas. Las columnas no
    numéricas se omiten. Los resultados se ordenan de mayor a menor según
    la cantidad de valores negativos.

    Parameters
    ----------
    base : pd.DataFrame
        DataFrame que contiene las observaciones que se desean analizar.

    Returns
    -------
    pd.DataFrame
        DataFrame con las columnas ``cantidad`` y ``porcentaje``.
    """
    numericas = base.select_dtypes(include="number")
    negativos = numericas.lt(0)
    return pd.DataFrame({
        "cantidad": negativos.sum(),
        "porcentaje": (negativos.mean() * 100).round(2),
    }).sort_values("cantidad", ascending=False)


def rows_with_negative_values(base):
    """
    Identifica las filas que contienen valores negativos en columnas
    numéricas de un DataFrame.

    Una fila se incluye si al menos uno de sus valores numéricos es menor
    que cero. Las columnas no numéricas no se consideran para la detección,
    pero se conservan en las filas devueltas.

    Parameters
    ----------
    base : pd.DataFrame
        DataFrame que contiene las observaciones que se desean analizar.

    Returns
    -------
    pd.DataFrame
        Filas originales que contienen al menos un valor numérico negativo.
    """
    numericas = base.select_dtypes(include="number")
    filas_negativas = numericas.lt(0).any(axis=1)
    return base.loc[filas_negativas]


def drop_missing_rows(base, umbral=None):
    """
    Evalúa la proporción global de valores faltantes de acuerdo a las
    siguientes condiciones:
    Si la proporción media de valores faltantes es menor o igual
    al umbral establecido, elimina las filas incompletas.
    En caso contrario, conserva el DataFrame original y no realiza
    ninguna eliminación, indicando que se requiere una revisión manual.

    Parameters
    ----------
    base : pd.DataFrame
        DataFrame que contiene las observaciones que se desean limpiar.
        Las filas con al menos un valor nulo se consideran incompletas.

    umbral : float, optional
        Proporción máxima global de valores faltantes permitida
        para ejecutar la eliminación de filas. Debe expresarse
        como una proporción entre 0 y 1.
        Si es None, utiliza el umbral definido en cfg.threshold_null

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        Tupla con dos DataFrames:

        - base_limpia : pd.DataFrame
            DataFrame resultante del proceso de limpieza. Si la
            proporción de valores faltantes no supera el umbral,
            contiene únicamente las filas completas. En caso
            contrario, conserva todas las observaciones originales.

        - filas_eliminadas : pd.DataFrame
            DataFrame con las filas que contenían al menos un valor
            nulo y fueron identificadas durante la evaluación.
            Si se supera el umbral, devuelve un DataFrame vacío
            porque no se elimina ninguna observación.
    """
    umbral = cfg.threshold_null if umbral is None else umbral
    filas = base[base.isna().any(axis=1)]
    if base.isna().mean().mean() <= umbral:
        return base.dropna(), filas
    print("El porcentaje de nulos supera el umbral: no se eliminan filas, revisar a mano.")
    return base, filas.iloc[0:0]





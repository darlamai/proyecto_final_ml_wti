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
    base = pd.read_csv(ruta)
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

def split_dataset(df, splits=None):
    """
    Divide un DataFrame con índice temporal en conjuntos de entrenamiento,
    validación y prueba, utilizando los intervalos de fechas definidos
    en la configuración del proyecto.

    La partición se realiza mediante selección de rangos temporales,
    conservando el orden cronológico de las observaciones y evitando
    la mezcla aleatoria de datos entre conjuntos.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame que contiene las observaciones que se desean particionar.
        Debe disponer de un índice temporal ordenado cronológicamente.

    splits : dict, optional
        Diccionario que define los intervalos temporales de cada partición.
        Cada clave identifica un conjunto (train, val o test) y su valor
        contiene las fechas inicial y final del intervalo correspondiente.
        Si es None o está vacío, utiliza los intervalos definidos
        en cfg.splits.

    Returns
    -------
    dict[str, pd.DataFrame]
        Diccionario cuyos valores son los DataFrames correspondientes
        a cada partición temporal. Las claves identifican los conjuntos
        de entrenamiento, validación y prueba.

        La selección de observaciones se realiza mediante rangos
        temporales inclusivos en ambos extremos.
    """
    splits = splits or cfg.splits
    return {k: df.loc[a:b] for k, (a, b) in splits.items()}

def validate_splits(df, partes, ventanas=None):
    """
    Verifica la consistencia temporal y la integridad de las particiones de entrenamiento, validación y prueba. 
    Esto garantiza mantener el orden cronológico, que no existan solapamientos temporales entre conjuntos, que 
    el número total de observaciones particionadas coincida con el intervalo temporal esperado y que las ventanas 
    de evaluación cubran el número de observaciones del conjunto de prueba.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame original utilizado para generar las particiones. Su índice temporal permite determinar el 
        intervalo global de observaciones que debe conservarse.

    partes : dict[str, pd.DataFrame]
        Diccionario que contiene los DataFrames correspondientes a las particiones temporales:
        - train: conjunto de entrenamiento.
        - val: conjunto de validación.
        - test: conjunto de prueba.

    ventanas : dict, optional
        Diccionario que define los intervalos temporales de las ventanas de evaluación del conjunto de prueba.
        Cada valor debe contener una fecha inicial y una fecha final.
        Si es None o está vacío, utiliza las ventanas definidas
        en cfg.test_windows.

    Returns
    -------
    bool
        Devuelve True si todas las verificaciones implementadas
        se cumplen satisfactoriamente.

    """

    ventanas = ventanas or cfg.test_windows

    assert partes["train"].index.max() < partes["val"].index.min()
    assert partes["val"].index.max() < partes["test"].index.min()

    a0, b0 = cfg.splits["train"][0], cfg.splits["test"][1]

    assert sum(len(v) for v in partes.values()) == len(df.loc[a0:b0])

    assert sum(len(df.loc[a:b]) for a, b in ventanas.values()) == len(partes["test"])

    return True
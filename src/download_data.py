################################################## DESCARGA DE DATOS #######################################################################
# Este script se encarga de descargar los datos necesarios para el proyecto. Es necesaria la conexión a internet para poder ejecutar este script correctamente.

from datetime import datetime
from pathlib import Path
import json

import pandas as pd
import yfinance as yf


# ============================================================
# 1. CONFIGURACIÓN
# ============================================================

start_date = "2010-01-01"

# Se fija como última fecha de descarga el último día del mes de Octubre 2026.
end_date = datetime(2026, 10, 31).strftime("%Y-%m-%d")

parameters = {
    "WTI": "CL=F",# precio del petróleo crudo de referencia de Estados Unidos WTI
}

# Campos de Yahoo que se guardan y nombre de la columna en el CSV.
# El cierre conserva el nombre "WTI" para no cambiar el resto del proyecto.
fields = {
    "Close": "WTI",       # precio de cierre
    "High": "WTI_High",   # precio máximo del día
    "Low": "WTI_Low",     # precio mínimo del día
}

project_root = Path(__file__).resolve().parents[1]
raw_dir = project_root / "data" / "raw"
raw_dir.mkdir(parents=True, exist_ok=True)




# ============================================================
# 2. DESCARGA
# ============================================================

def download_prices() -> pd.DataFrame:

    """Descarga el cierre, el máximo y el mínimo diarios del precio del petróleo WTI."""

    downloaded = yf.download(
        tickers=list(parameters.values()),
        start=start_date,
        end=end_date,
        interval="1d",
        auto_adjust=False,
        actions=False,
        group_by="column",
        progress=True,
        threads=True,
    )


    if downloaded.empty:
        raise RuntimeError(
            "La descarga no devolvió datos. "
            "Verifica tu conexión y los símbolos."
        )

    ticker = parameters["WTI"]
    columnas = {}

    # Extraer Close, High y Low, sin utilizar Open ni Volume.
    for field, nombre in fields.items():

        if isinstance(downloaded.columns, pd.MultiIndex):
            if field not in downloaded.columns.get_level_values(0):
                raise ValueError(f"No se encontró la columna {field}.")

            data = downloaded.xs(
                field,
                axis=1,
                level=0,
            )
            serie = data[ticker] if ticker in data.columns else data.iloc[:, 0]
        else:
            # Protección por si Yahoo devuelve una sola serie.
            if field not in downloaded.columns:
                raise ValueError(f"No se encontró la columna {field}.")

            serie = downloaded[field]

        columnas[nombre] = serie.astype(float)

    prices = pd.DataFrame(columnas)

    # Comprobar que estén presentes todas las columnas
    missing_columns = set(fields.values()) - set(prices.columns)

    if missing_columns:
        raise ValueError(
            f"Faltan columnas en la descarga: {sorted(missing_columns)}"
        )

    prices = prices[list(fields.values())].sort_index()
    prices.index = pd.to_datetime(prices.index)
    prices.index.name = "Date"

    return prices




# ============================================================
# 3. GUARDAR DATOS Y METADATOS
# ============================================================

def main() -> pd.DataFrame:

    # --------------------------------------------------------
    # 1. Buscar archivos CSV existentes
    # --------------------------------------------------------

    existing_files = sorted(
        raw_dir.glob("wti_market_data_*.csv"),
        key=lambda x: x.stat().st_mtime,
        reverse=True,
    )

    # --------------------------------------------------------
    # 2. Si existe un archivo con todas las columnas, cargarlo y evitar descarga
    # --------------------------------------------------------

    if existing_files:

        csv_path = existing_files[0]

        prices = pd.read_csv(
            csv_path,
            index_col="Date",
            parse_dates=True,
        )

        faltan = set(fields.values()) - set(prices.columns)

        if not faltan:

            prices.index.name = "Date"

            print("\nYa existe una descarga previa.")
            print(f"Leyendo archivo: {csv_path.name}")
            print("\nDatos cargados correctamente.")
            print(f"Observaciones: {len(prices)}")
            print(f"Primera fecha: {prices.index.min().date()}")
            print(f"Última fecha: {prices.index.max().date()}")
            print(f"\nArchivo utilizado: {csv_path}")
            print("\nValores faltantes por serie:")
            print(prices.isna().sum())

            return prices

        # El archivo existente es de una versión anterior (solo cierre): se descarga de nuevo.
        print(f"\nEl archivo {csv_path.name} no tiene las columnas {sorted(faltan)}.")
        print("Se descargará una nueva versión con cierre, máximo y mínimo.")

    # --------------------------------------------------------
    # 3. Si no existe archivo válido, descargar datos
    # --------------------------------------------------------

    else:
        print("\nNo existen archivos previos. Iniciando descarga...")

    prices = download_prices()

    download_date = datetime.now().strftime("%Y%m%d")

    csv_path = raw_dir / (
        f"wti_market_data_{download_date}.csv"
    )

    metadata_path = raw_dir / (
        f"wti_market_metadata_{download_date}.json"
    )

    # Si ya existe un archivo con la fecha de hoy (versión anterior), se conserva con otro nombre.
    if csv_path.exists():
        respaldo = csv_path.with_name("respaldo_solo_cierre_" + csv_path.name)
        csv_path.rename(respaldo)
        print(f"Archivo anterior renombrado a: {respaldo.name}")

    # --------------------------------------------------------
    # 4. Guardar CSV
    # --------------------------------------------------------

    prices.to_csv(
        csv_path,
        index=True,
        float_format="%.10g",
    )

    # --------------------------------------------------------
    # 5. Generar metadatos
    # --------------------------------------------------------

    metadata = {
        "source": "Yahoo Finance",
        "download_timestamp": datetime.now().isoformat(),
        "start_date_requested": start_date,
        "end_date_exclusive": end_date,
        "interval": "1d",
        "price_field": list(fields.keys()),
        "column_names": fields,
        "auto_adjust": False,
        "tickers": parameters,
        "columns": list(prices.columns),
        "rows": int(len(prices)),
        "first_observation": str(prices.index.min().date()),
        "last_observation": str(prices.index.max().date()),
        "missing_values_by_column": {
            col: int(prices[col].isna().sum())
            for col in prices.columns
        },
        "note": (
            "Datos originales descargados. "
            "No se han imputado faltantes ni calculado retornos."
        ),
    }

    metadata_path.write_text(
        json.dumps(metadata, indent=4, ensure_ascii=False),
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # 6. Mostrar resultados
    # --------------------------------------------------------

    print("\nDescarga completada.")
    print(f"Observaciones: {len(prices)}")
    print(f"Primera fecha: {prices.index.min().date()}")
    print(f"Última fecha: {prices.index.max().date()}")
    print(f"\nDatos guardados en: {csv_path}")
    print(f"Metadatos guardados en: {metadata_path}")
    print("\nValores faltantes por serie:")
    print(prices.isna().sum())

    return prices


if __name__ == "__main__":
    main()

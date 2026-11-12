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

project_root = Path(__file__).resolve().parents[1]
raw_dir = project_root / "data" / "raw"
raw_dir.mkdir(parents=True, exist_ok=True)




# ============================================================
# 2. DESCARGA
# ============================================================

def download_close_prices() -> pd.DataFrame:

    """Descarga el cierre diario del precio del petróleo WTI."""

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

    # Extraer Close, sin utilizar High, Low, Open o Volume.
    if isinstance(downloaded.columns, pd.MultiIndex):
        if "Close" not in downloaded.columns.get_level_values(0):
            raise ValueError("No se encontró la columna Close.")

        close = downloaded.xs(
            "Close",
            axis=1,
            level=0,
        )
    else:
        # Protección por si Yahoo devuelve una sola serie.
        if "Close" not in downloaded.columns:
            raise ValueError("No se encontró la columna Close.")

        close = downloaded[["Close"]].copy()
        close.columns = [list(parameters.keys())[0]]

    # Convertir símbolos de Yahoo a nombres legibles.
    rename_map = {
        ticker: name
        for name, ticker in parameters.items()
    }
    close = close.rename(columns=rename_map)

    # Comprobar que esté presente la serie WTI 
    missing_columns = set(parameters.keys()) - set(close.columns)

    if missing_columns:
        raise ValueError(
            f"Falta la serie WTI en la descarga: {sorted(missing_columns)}"
        )

    close = close[list(parameters.keys())].sort_index()
    close.index = pd.to_datetime(close.index)
    close.index.name = "Date"

    return close



# ============================================================
# 3. GUARDAR DATOS Y METADATOS
# ============================================================

def main() -> None:
    close = download_close_prices()

    download_date = datetime.now().strftime("%Y%m%d")

    csv_path = raw_dir / (
        f"wti_market_data_{download_date}.csv"
    )

    metadata_path = raw_dir / (
        f"wti_market_metadata_{download_date}.json"
    )

    # Evitar sobrescribir silenciosamente una descarga previa.
    if csv_path.exists() or metadata_path.exists():
        raise FileExistsError(
            f"Ya existe una descarga para {download_date}. "
            "Cambia el nombre o guarda una nueva versión."
        )

    close.to_csv(csv_path, index=True, float_format="%.10g")

    metadata = {
        "source": "Yahoo Finance",
        "download_timestamp": datetime.now().isoformat(),
        "start_date_requested": start_date,
        "end_date_exclusive": end_date,
        "interval": "1d",
        "price_field": "Close",
        "auto_adjust": False,
        "tickers": parameters,
        "columns": list(close.columns),
        "rows": int(len(close)),
        "first_observation": str(close.index.min().date()),
        "last_observation": str(close.index.max().date()),
        "missing_values_by_column": {
            col: int(close[col].isna().sum())
            for col in close.columns
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

    print("\nDescarga completada.")
    print(f"Observaciones: {len(close)}")
    print(f"Primera fecha: {close.index.min().date()}")
    print(f"Última fecha: {close.index.max().date()}")
    print(f"\nDatos guardados en: {csv_path}")
    print(f"Metadatos guardados en: {metadata_path}")
    print("\nValores faltantes por serie:")
    print(close.isna().sum())



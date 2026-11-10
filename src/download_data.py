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

# Yahoo Finance considera end_date como fecha exclusiva y se utiliza la fecha actual de ejecución como límite.
end_date = datetime.now().strftime("%Y-%m-%d")

parameters = {
    "WTI": "CL=F",# precio del petróleo crudo de referencia de Estados Unidos WTI
    "VIX": "^VIX",# indicador de  volatilidad del mercado de  opciones del S&P 500 (inversores esperan que el mercado sea más volátil en el futuro- próximos 30 días)
    "DXY": "DX-Y.NYB", # índice del dólar estadounidense (DXY) mide el valor del dólar frente a una cesta de seis monedas extranjeras principales
    "GOLD": "GC=F", # precio futuro del oro y se expresa en dólares por onza troy
    "TNX": "^TNX", # rendimiento del bono del Tesoro de EE.UU, mide la tasa de interés del bono del Tesoro a 10 años, que es un indicador clave de las tasas de interés a largo plazo en la economía estadounidense
    "SP500": "^GSPC", # índice bursátil que representa el rendimiento de las 500 empresas más grandes que cotizan en bolsa en Estados Unidos
    "NG": "NG=F",# precio futuro del gas natural
}

project_root = Path(__file__).resolve().parents[2]
raw_dir = project_root / "data" / "raw"
raw_dir.mkdir(parents=True, exist_ok=True)




# ============================================================
# 2. DESCARGA
# ============================================================

def download_close_prices() -> pd.DataFrame:

    """Descarga los cierres diarios de los activos seleccionados."""

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

    # Comprobar que estén presentes las siete series.
    missing_columns = set(parameters.keys()) - set(close.columns)

    if missing_columns:
        raise ValueError(
            f"Faltan series en la descarga: {sorted(missing_columns)}"
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


if __name__ == "__main__":
    main()


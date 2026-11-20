# =============================================================================
# src/config.py
# Configuración central del proyecto: rutas, columnas de la base, cortes
# temporales (desarrollo, test y tramo de diagnóstico), transformación de las
# variables y parámetros de limpieza. Todos los notebooks importan este módulo
# para usar exactamente las mismas fechas y parámetros.
#

# Uso de asistentes de IA (requisito de la rúbrica):
#   Partes de este código se generaron o adaptaron con ayuda de un asistente de IA: Claude
# =============================================================================



from pathlib import Path

root = Path(__file__).resolve().parents[1]
raw_csv = root / "data" / "raw" 
processed = root / "data" / "processed"


columns= ["WTI","WTI_High", "WTI_Low"]



#   Prompt (Claude, 11/12-nov-2026): "Ayúdame a definir train, test y validation
#   adecuadamente: asi tengo dividida mi data: train: ("2010-01-04", "2018-12-31"),
#   val: ("2019-01-01", "2019-12-31"), test: ("2020-01-01", "2026-10-30")"

DEV_INI, DEV_FIN   = "2010-01-01", "2024-12-31"
TEST_INI, TEST_FIN = "2025-01-01", "2026-10-31"

splits = {
    "dev":     ("2010-01-01", "2024-12-31"),   # train + validación (folds walk-forward)
    "test":    ("2025-01-01", "2026-10-31"),   # test final
    "eda_fin": "2019-06-30",                   # fin del tramo de diagnóstico (train más corto)
}

transf = {
    "WTI":   "logret"
}
threshold_null = 0.05      # se eliminan filas con nulos si el % medio de nulos es <= 5 %
threshold_outlier_z = 5     # |z| > 5, con la desviación calculada solo en train

# Propuestos: confirmar con el tutor y dejar escritos ANTES de ver resultados
k_grid = list(range(1, 21))          # longitud de la ventana de entrada
h_grid = [1, 5, 10, 21, 63]          # horizonte de predicción / generación


parkinson = {"WTI_park": ("WTI_High", "WTI_Low")}
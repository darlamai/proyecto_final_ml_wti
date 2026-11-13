from pathlib import Path

root = Path(__file__).resolve().parents[1]
raw_csv = root / "data" / "raw" 
processed = root / "data" / "processed"


columns= ["WTI","WTI_High", "WTI_Low"]

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
from pathlib import Path

root = Path(__file__).resolve().parents[1]
raw_csv = root / "data" / "raw" / "wti_market_data_20261021.csv"
processed = root / "data" / "processed"


target = "WTI"
ex = ["VIX", "DXY", "GOLD", "TNX", "SP500", "NG"]
columns = [target] + ex

splits = {
    "train": ("2010-01-04", "2018-12-31"),
    "val":   ("2019-01-01", "2019-12-31"),
    "test":  ("2020-01-01", "2026-10-20"),
}

test_windows = {
    "covid":       ("2020-01-01", "2020-12-31"),
    "rec_2021":    ("2021-01-01", "2021-12-31"),
    "rob_2022":    ("2022-01-01", "2022-12-31"),
    "calma_23_25": ("2023-01-01", "2025-12-31"),
    "rec_2026":    ("2026-01-01", "2026-10-20"),
}

transf = {
    "WTI":   "logret",
    "VIX":   "logret",
    "DXY":   "logret",
    "GOLD":  "logret",
    "SP500": "logret",
    "NG":    "logret",
    "TNX":   "diff",     # tasa de interés: el cambio natural es en puntos
}
threshold_null = 0.05      # se eliminan filas con nulos si el % medio de nulos es <= 5 %
threshold_outlier_z = 5     # |z| > 5, con la desviación calculada solo en train

# Propuestos: confirmar con el tutor y dejar escritos ANTES de ver resultados
k_grid = list(range(1, 21))          # longitud de la ventana de entrada
h_grid = [1, 5, 10, 21, 63]          # horizonte de predicción / generación
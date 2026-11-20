# Pronóstico de la volatilidad diaria del petróleo WTI: EGARCH vs. Random Forest

**Proyecto final — Aprendizaje Supervisado · Maestría en Inteligencia Artificial (USFQ)**
**Autora:** Darlyn Ludeña

Este proyecto compara un modelo econométrico y un modelo de machine learning para pronosticar la **volatilidad diaria del petróleo WTI**, y evalúa si la diferencia entre ambos es estadísticamente significativa.

| Modelo | Descripción |
|---|---|
| **Econométrico (base)** | EGARCH(1,1,1) con media constante y errores t de Student |
| **Machine learning** | Random Forest con rezagos de medidas de volatilidad, selección de variables por información mutua y optimización con Optuna |

---

## Pregunta de investigación

> ¿Un Random Forest basado en rezagos de medidas de volatilidad pronostica la volatilidad diaria del WTI con mayor precisión que un EGARCH(1,1,1) con media constante y errores t de Student, y es esa diferencia estadísticamente significativa?

- $H_0$: ambos modelos tienen la misma precisión de pronóstico de la volatilidad del día siguiente.
- $H_1$: la precisión de ambos modelos es distinta.

---

## Datos

- **Fuente:** Yahoo Finance, ticker `CL=F` (serie continua de futuros WTI), descargada con `yfinance`.
- **Periodo:** 4 de enero de 2010 – 30 de octubre de 2026 (4.233 sesiones).
- **Columnas:** precio de cierre (`WTI`), máximo (`WTI_High`) y mínimo (`WTI_Low`) de cada sesión.
- **Limpieza:** se eliminan las sesiones del 20 y 21 de abril de 2020, con precios negativos (vencimiento del contrato de mayo de 2020), porque el logaritmo no está definido para precios ≤ 0.

**Variables construidas:**

- **Retornos logarítmicos en %:** $r_t = 100 \cdot \ln(P_t / P_{t-1})$
- **Estimador de volatilidad de Parkinson en %²** (proxy de la varianza diaria y variable objetivo):

$$\sigma^2_{P,t} = \frac{\left[100 \cdot \ln(H_t / L_t)\right]^2}{4 \ln 2}$$

**División temporal:**

| Conjunto | Periodo | Observaciones | Uso |
|---|---|---|---|
| Desarrollo | ene 2010 – dic 2024 | 3.769 | Entrenamiento y validación en 10 folds |
| Tramo de diagnóstico | ene 2010 – jun 2019 | 2.385 | EDA y decisiones del modelo base (entrenamiento común a todos los folds) |
| Test final | ene 2025 – oct 2026 | 461 | Evaluación final, usada una sola vez |

La validación usa **Time Series Cross-Validation** con ventana expansiva: 2 repeticiones × 5 folds, con periodos de validación de 12 meses. No se usa k-fold aleatorio porque introduciría información futura en el entrenamiento.

---

## Estructura del repositorio

```
proyecto_final_ml_wti/
├── Proyecto_ML_WTI.ipynb          # Notebook principal: introducción, metodología, ejecución y conclusiones
├── README.md
├── requirements.txt
├── data/
│   ├── raw/                       # Datos descargados de Yahoo Finance (CSV + metadatos JSON)
│   └── processed/                 # Bases procesadas, folds, features y resultados de la selección
├── notebooks/
│   ├── 01.eda_raw.ipynb           # Calidad de datos, transformación y división temporal
│   ├── 02.eda_returns.ipynb       # Diagnóstico de los retornos (EDA)
│   ├── 03.modelo_base.ipynb       # Modelo econométrico EGARCH con media constante
│   ├── 04.feature_extraction.ipynb# Construcción de 67 variables por rezagos
│   ├── 05.feature_selection.ipynb # Selección tipo filter (información mutua) y t-SNE
│   ├── 06.random_forest.ipynb     # Pipeline, optimización con Optuna y diagnóstico
│   └── 07.comparison_model.ipynb  # Comparación estadística y test final
├── src/
│   ├── config.py                  # Rutas, columnas, fechas de corte y parámetros
│   ├── download_data.py           # Descarga de precios desde Yahoo Finance
│   ├── load_clean.py              # Carga y limpieza de los datos crudos
│   ├── preprocessing.py           # Retornos logarítmicos y estimador de Parkinson
│   ├── eda_utils.py               # Funciones del EDA y de los folds
│   ├── ar_garch.py                # Diagnóstico y pronóstico del EGARCH
│   ├── feature_extraction.py      # Variables por rezagos
│   ├── random_forest.py           # Pipeline, pérdidas y función objetivo de Optuna
│   └── comparison.py              # Pérdidas, Wilcoxon, t pareada y Diebold-Mariano
├── results/                       # Errores por fold, pronósticos, estudio de Optuna y comparación
└── figures/                       # Gráficos generados por los notebooks
```

---

## Metodología

| Etapa | Notebook | Qué se hace |
|---|---|---|
| **EDA de los datos crudos** | `01.eda_raw` | Faltantes, valores no positivos, días sin variación, huecos en el calendario, saltos extremos; cálculo de retornos y Parkinson; división temporal y folds |
| **EDA de los retornos** | `02.eda_returns` | Estadística descriptiva y Jarque-Bera; estacionariedad (ADF, PP, KPSS); estacionalidad (Kruskal-Wallis); dependencia en la media (ACF, PACF, Ljung-Box) y en la varianza (ACF de $r_t^2$, ARCH-LM) |
| **Modelo base** | `03.modelo_base` | Comparación de tres especificaciones EGARCH(1,1,1)-t; verificación de supuestos sobre los residuos estandarizados; pronóstico a un paso en los 10 folds |
| **Feature extraction** | `04.feature_extraction` | 67 variables con información hasta τ−1: 20 rezagos de $r^2$, $\lvert r\rvert$ y Parkinson; volatilidad móvil y Parkinson promedio en 5, 21 y 63 días; retorno con signo |
| **Feature selection y t-SNE** | `05.feature_selection` | Selección tipo *filter* con `SelectKBest` e información mutua dentro del pipeline; elección de k con validación temporal; visualización t-SNE coloreada por el nivel de Parkinson |
| **Random Forest** | `06.random_forest` | `Pipeline` de scikit-learn (selección + Random Forest); optimización con Optuna (TPE y poda por mediana); curvas de entrenamiento y validación, curva de aprendizaje e importancia de variables |
| **Comparación** | `07.comparison_model` | Wilcoxon y t pareada sobre los 10 folds; Diebold-Mariano con varianza Newey-West sobre los pronósticos diarios; referencia ingenua (media de Parkinson de 21 días); evaluación en el test final |

**Funciones de pérdida:** QLIKE $= y/h + \ln h$ (métrica principal, robusta al ruido del proxy) y MSE (complementaria), donde $y$ es el Parkinson observado y $h$ la varianza pronosticada.

---

## Resultados principales

| | EGARCH media const. | Random Forest | Ingenuo (media 21 días) |
|---|---|---|---|
| QLIKE validación (media 10 folds) | 2,866 | 2,849 | 2,829 |
| QLIKE test 2025–2026 | 3,239 | **2,839** | 2,974 |
| MSE validación (media) | **989,2** | 1.324,4 | 1.128,9 |
| MSE test | 815,4 | **729,4** | 784,2 |
| Folds con menor QLIKE (EGARCH vs. RF) | 3 de 10 | 7 de 10 | — |

| Prueba | Resultado |
|---|---|
| Wilcoxon (10 folds) | QLIKE *p* = 0,492; MSE *p* = 0,084 |
| Diebold-Mariano (validación, por repetición) | QLIKE *p* = 0,724 y 0,853; MSE *p* = 0,092 y 0,089 |
| Diebold-Mariano (test) | QLIKE *p* = 0,147; MSE *p* = 0,722 |

**Conclusión:** no se rechaza $H_0$. El Random Forest obtuvo pérdidas ligeramente menores en validación y en el test, y su error de test fue casi igual al de validación, pero ninguna diferencia es estadísticamente significativa al 5 %. En validación, el Random Forest fue mejor en los periodos de volatilidad moderada y el EGARCH en los folds de 2020; en el test, el Random Forest reaccionó mejor al pico de marzo de 2026. La mayor parte de la información predecible está en la persistencia de la volatilidad, que también capta un promedio móvil simple.

---

## Cómo ejecutar el proyecto

### Opción 1: local

```bash
git clone https://github.com/darlamai/proyecto_final_ml_wti.git
cd proyecto_final_ml_wti

python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate

pip install -r requirements.txt
```

1. Descargar los datos (solo la primera vez; requiere internet):
   ```bash
   python -m src.download_data
   ```
2. Abrir `Proyecto_ML_WTI.ipynb` **desde la raíz del proyecto** y ejecutarlo completo. El notebook principal ejecuta los siete notebooks en orden con `jupyter nbconvert --execute`, y cada uno guarda sus resultados en `data/processed/` o `results/` para el siguiente.

También se puede ejecutar cada notebook de `notebooks/` por separado, siempre en orden del 01 al 07.

### Opción 2: Google Colab

En la primera celda de un notebook nuevo de Colab:

```python
!git clone https://github.com/darlamai/proyecto_final_ml_wti.git
%cd proyecto_final_ml_wti
!pip install -r requirements.txt
!python -m src.download_data
```

Después abrir y ejecutar `Proyecto_ML_WTI.ipynb`.

> **Nota:** el estudio de Optuna se guarda en `results/optuna_rf.db`. Si el archivo existe, el notebook 06 lo carga y no repite las 50 pruebas. Para repetir la búsqueda desde cero, elimine ese archivo.

---

## Dependencias

Python 3.10 o superior. Principales librerías:

| Librería | Uso |
|---|---|
| `numpy`, `pandas` | Manipulación de datos |
| `matplotlib`, `plotly` | Gráficos |
| `scipy` | Pruebas estadísticas (Wilcoxon, Shapiro-Wilk, Kruskal-Wallis, Jarque-Bera) |
| `statsmodels` | ADF, KPSS, ACF/PACF, Ljung-Box, ARCH-LM |
| `arch` | Modelo EGARCH y prueba de Phillips-Perron |
| `scikit-learn` | Pipeline, `SelectKBest`, información mutua, Random Forest, t-SNE |
| `optuna` | Optimización de hiperparámetros |
| `yfinance` | Descarga de datos |
| `joblib` | Guardado del estudio de Optuna |

El archivo `requirements.txt` contiene las versiones exactas usadas.

---

## Uso de asistentes de IA

Parte del código se generó o adaptó con ayuda de un asistente de IA (Claude, Anthropic). Como pide la rúbrica, los prompts utilizados están como comentarios de Python sobre cada función en los módulos de `src/`.

---

## Referencias principales

[1] R. F. Engle, "Autoregressive conditional heteroscedasticity with estimates of the variance of United Kingdom inflation," *Econometrica*, vol. 50, no. 4, pp. 987–1007, 1982.

[2] D. B. Nelson, "Conditional heteroskedasticity in asset returns: A new approach," *Econometrica*, vol. 59, no. 2, pp. 347–370, 1991.

[3] L. Breiman, "Random forests," *Mach. Learn.*, vol. 45, no. 1, pp. 5–32, 2001.

[4] M. Parkinson, "The extreme value method for estimating the variance of the rate of return," *J. Business*, vol. 53, no. 1, pp. 61–65, 1980.

[5] A. J. Patton, "Volatility forecast comparison using imperfect volatility proxies," *J. Econometrics*, vol. 160, no. 1, pp. 246–256, 2011.

[6] F. X. Diebold y R. S. Mariano, "Comparing predictive accuracy," *J. Bus. Econ. Statist.*, vol. 13, no. 3, pp. 253–263, 1995.

[7] T. Akiba, S. Sano, T. Yanase, T. Ohta y M. Koyama, "Optuna: A next-generation hyperparameter optimization framework," en *Proc. 25th ACM SIGKDD*, 2019, pp. 2623–2631.

[8] F. Pedregosa *et al.*, "Scikit-learn: Machine learning in Python," *J. Mach. Learn. Res.*, vol. 12, pp. 2825–2830, 2011.

Las referencias completas de cada etapa se encuentran al final de cada notebook.

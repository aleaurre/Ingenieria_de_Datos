---
title: "05 — Missing Data Detective (Ames Housing)"
date: 2025-09-15
number: 5
status: "En progreso"
tags: [Calidad de datos, Missingness, MCAR/MAR/MNAR, Imputación, Outliers, Anti-leakage, Ames Housing]
notebook: —
drive_viz: —
dataset: "Ames Housing — Kaggle"
time_est: "3 h 00 m"
time_spent: "—"
---

# {{ page.meta.title }}

<span class="pill">{{ page.meta.status }}</span>
<span class="pill">#{{ page.meta.number }}</span>
{% if page.meta.tags %}{% for t in page.meta.tags %}<span class="pill">{{ t }}</span>{% endfor %}{% endif %}

!!! abstract "Resumen ejecutivo"
    **Objetivo:** auditar faltantes, clasificar **MCAR/MAR/MNAR**, detectar **outliers (IQR/Z-score)**, **imputar** de forma informada y construir un **pipeline reproducible sin leakage**.  
    **Scope:** dataset **Ames Housing** (tabular, numéricas + categóricas).  
    **Entregables:** notebook con pasos y visuales, resumen de decisiones (ADR-lite) y pipeline `ColumnTransformer` + `SimpleImputer` listo para integrar a modelado.  
    **Guía base:** estructura de pasos de la consigna oficial (Parte 1 y Parte B).

**Enlaces rápidos:**  
[Consigna Práctica 5](https://juanfkurucz.com/ucu-id/ut2/05-missing-data-detective/) · [Pandas docs](https://pandas.pydata.org/) · [scikit-learn: Imputation](https://scikit-learn.org/) 

---

## Contexto
Los **faltantes** y **outliers** afectan la validez estadística y el desempeño de modelos. Esta práctica guía un flujo **auditar → diagnosticar → imputar → validar impacto**, cerrando con **anti-leakage** y **pipeline reproducible**. La consigna sugiere además reflexiones éticas y de transparencia. 

## Objetivos
- [x] Cuantificar *missingness* por columna/fila y visualizar patrones.  
- [x] Clasificar **MCAR/MAR/MNAR** con evidencia.  
- [x] Detectar **outliers** con IQR y Z-score y comparar.
- [x] Aplicar **imputación** acorde al tipo (media/mediana/moda/constante).  
- [x] Implementar **anti-leakage** y **pipeline** con `ColumnTransformer`.   
- [x] Responder preguntas de **Parte B** (impacto, ética, reproducibilidad). 

---

## Actividades (con tiempos estimados)

| Actividad | Estimado | Real | Nota |
|---|---:|---:|---|
| Setup + carga Ames | 10 m | **7 m** | Descargar/leer CSV; `df.info()`, `df.dtypes`. |
| Crear *missing* sintético (MCAR/MNAR/MAR) | 10 m | **8 m** | Semilla fija; reglas por columna. |
| Auditoría inicial | 20 m | **14 m** | % faltantes por columna/fila; duplicados; memoria. |
| Patrones de *missing* | 20 m | **15 m** | Barras Top-N, histograma por fila; co-faltantes. |
| Clasificación MCAR/MAR/MNAR | 15 m | **10 m** | Tablas por grupo (`Neighborhood`, `Garage Type`). |
| Outliers IQR/Z-score | 20 m | **14 m** | Límites, conteos, comparación métodos. |
| Imputación base (num/cat) | 20 m | **12 m** | Mediana/media; moda/“Unknown”; *flags* opcionales. |
| Anti-leakage + splits | 15 m | **10 m** | `fit` en **train**; `transform` en valid/test. |
| Pipeline reproducible | 20 m | **14 m** | `ColumnTransformer` + `SimpleImputer` + `OneHotEncoder`. |
| Análisis de impacto | 20 m | **12 m** | Antes vs después: shape, cardinalidad, outliers. |

> **Totales** — Estimado: **2 h 50 m** · Real: **1 h 56 m** · Δ: **−54 m** (**−31.8%**).

---

## Desarrollo

### 1) Carga y chequeos básicos
- `df.shape`, `df.info()`, `df.isna().sum()`, `df.duplicated().sum()`, `df.memory_usage(deep=True)`.  
- Mini **data dictionary** (nombre, unidad/tipo, semántica). 

### 2) *Missingness* sintético (para practicar diagnósticos)
- **MCAR** (p.ej., `Year Built` ~8% aleatorio).  
- **MAR** (p.ej., `Garage Area` condicionado a `Garage Type`).  
- **MNAR** (p.ej., *truncation* en colas altas de `SalePrice`).  
Documentar la lógica y sembrar `np.random.seed(42)`. 

### 3) Auditoría y patrones
- **Por columna:** conteo y %; Top-N con barra ordenada.  
- **Por fila:** histograma de #faltantes.  
- Opcional: matriz de co-faltantes (sin *missingno*, con `pandas`/`matplotlib`). 

### 4) Clasificar MCAR/MAR/MNAR (evidencia mínima)
- **MCAR:** independencia respecto de otras columnas/grupos.  
- **MAR:** asociación con variables observadas (tablas por grupo).  
- **MNAR:** patrón ligado al propio valor (e.g., colas altas/altas tasas). 
### 5) Outliers
- **IQR:** límites `[Q1−1.5·IQR, Q3+1.5·IQR]`.  
- **Z-score:** |z| > 3 si distribución ~normal.  
- Comparar conteos y *overlap* entre métodos; decidir estrategia (capping, winsorizing, etiqueta, o dejar). 

### 6) Imputación (base y por tipo)
- **Numéricas:** `median` si asimetría/colas; `mean` si ~normal.  
- **Categóricas:** `most_frequent` o constante `"Unknown"` (+ *flags* si sospecha MNAR).  
- Registrar *rationale* por columna. 

### 7) Anti-leakage y *splits*
- Dividir **antes** de imputar (`train/valid/test`); **fit** imputer con **train** y **transform** en valid/test.  
- Separar columnas num/cat y reutilizar en el pipeline. 

### 8) Pipeline reproducible
- `ColumnTransformer([('num', SimpleImputer(strategy='median'), num_cols), ('cat', SimpleImputer(strategy='most_frequent'), cat_cols)])` + `OneHotEncoder`/`StandardScaler` si aplica.  
- Probar `preprocessor.fit_transform(df)` y registrar `shape`/tipo de salida. 

### 9) Análisis de impacto (antes vs después)
- Shape, % faltantes → 0, #categorías tras OHE, *drift* en medias/medianas, outliers residuales.  
- Notas sobre **bias** potencial introducido por la imputación. 


<details class="md-details">
  <summary><strong>Paso a paso (ejecución)</strong></summary>
  <ol>
    <li><strong>Setup:</strong> Importar `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`; fijar semilla. </li>
    <li><strong>Cargar Ames</strong> Leer CSV; `info()`, `describe()`, `dtypes`, memoria.</li>
    <li><strong>Crear faltantes sintéticos</strong> MCAR/MAR/MNAR con `np.nan`.</li>
    <li><strong>Patrones</strong> Barra Top-N % por columna + histograma por fila.</li>
    <li><strong>Clasificación</strong> Tablas por grupo (`groupby`) y descriptivos en subconjuntos. </li>
    <li><strong>Outliers</strong> Funciones IQR y Z-score; comparar por columna.</li>
    <li><strong>Imputación base</strong> Num = mediana; cat = moda/constante; *flags* opcionales.  </li>
    <li><strong>Anti-leakage</strong>: `train_test_split`; `SimpleImputer.fit(X_train)` → `transform` resto. </li>
    <li><strong>Pipeline</strong> `ColumnTransformer` (+ `OneHotEncoder`/`StandardScaler` si es necesario).  </li>
    <li><strong>Impacto</strong>: Métricas “antes vs después”, guardar figuras/tablas a Drive.</li>
  </ol>
</details>



---

## Métricas / Indicadores de calidad

## Métricas / Indicadores de calidad

| Indicador | Valor/Observación |
|---|---|
| Columnas con *missing* | 19 / ~15% (Top-N: `Garage Type`, `BsmtQual`, `FireplaceQu`) |
| Faltantes por fila (p95/p99) | 6 / 11 |
| Outliers IQR (columna con más casos) | `Lot Area` ≈ 127 outliers |
| Outliers Z-score (|z|>3) | `SalePrice` ≈ 65 casos; `Garage Area` ≈ 42 |
| Estrategias de imputación aplicadas | num: **mediana** / **media** según distribución; cat: **moda** o constante `"Unknown"` |
| Post-pipeline: filas × columnas | 1460 × ~240 (tras OneHotEncoder) |
| Verificación anti-leakage | ✅ imputers *fit* en train; *transform* en valid/test |

!!! tip "Criterios de aceptación"
    - [x] *Missingness* cuantificado y visualizado.  
    - [x] Clasificación **MCAR/MAR/MNAR** con evidencia mínima.  
    - [x] **IQR** y **Z-score** comparados y justificados.  
    - [x] **Imputación** consistente por tipo + notas de sesgo.  
    - [x] **Pipeline reproducible** sin leakage y probado.  

---

## Decisiones clave (ADR-lite)
- **Flags de *missing***: añadir indicadores en columnas **sospecha MNAR**; no en MCAR bajo.  
- **Imputación numérica**: **mediana** por robustez; revisar normalidad para `mean`.  
- **Categóricas**: `most_frequent` y constante `"Unknown"` donde pérdida de semántica sea tolerable.  
- **Outliers**: **cap** a límites IQR para columnas con fuerte asimetría; **Z-score** sólo en ~normal.  
- **Reproducibilidad**: consolidar en `ColumnTransformer` para evitar divergencias entre EDA y producción.

!!! warning "Riesgos / Supuestos"
    - **Leakage**: imputación/escala ajustada con todo el dataset. → *Mitigación*: `fit` solo en **train**. 
    - **Sesgo por imputación**: medias arrastran colas; constantes agregan clases espurias.  
    - **Outliers verdaderos**: riesgo de eliminar señal de negocio. → *Mitigación*: contrastar con dominio.

---

## Evidencias
- **Notebook:** [docs/evidencias/Practica5.ipynb](../../evidencias/Aurrecochea-Práctica5.ipynb)
- **Visualizaciones:**

<div class="cards-grid media">

  <div class="card">
    <img src="../../assets/Práctica5/missing_patterns.png" alt="Patrones de missingness por columna y co-faltantes" loading="lazy">
    <div class="caption">
      Patrones de <em>missingness</em>
      <small>Auditoría general de nulos</small>
    </div>
  </div>

  <div class="card">
    <img src="../../assets/Práctica5/distribution_comparison.png" alt="Comparación de distribuciones antes vs después de imputar" loading="lazy">
    <div class="caption">
      Distribuciones: antes vs después
      <small>Impacto directo de la imputación</small>
    </div>
  </div>

  <div class="card">
    <img src="../../assets/Práctica5/outliers_analysis.png" alt="Análisis de outliers por IQR y Z-score" loading="lazy">
    <div class="caption">
      Outliers (IQR/Z-score)
      <small>Decisión basada en distribución</small>
    </div>
  </div>

  <div class="card">
    <img src="../../assets/Práctica5/correlation_comparison.png" alt="Mapa de correlaciones antes y después del preprocesamiento" loading="lazy">
    <div class="caption">
      Correlación: antes vs después
      <small>Chequeo de drift estructural</small>
    </div>
  </div>

</div>



---

## Reproducibilidad
Entorno sugerido: `python 3.11`; `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `scipy`.  
Checklist:
1) Semilla fija (`np.random.seed`).  
2) **Split** antes de imputar (anti-leakage). 
3) Pipeline único (`ColumnTransformer` + `SimpleImputer` + `OneHotEncoder`).  
4) Guardar artefactos a `docs/assets/Practica5/` y Drive.

---

!!! note "Reflexión"
    El análisis visual evidencia que los patrones de missingness no son aleatorios: algunas columnas presentan ausencias sistemáticas que se asocian a características específicas (p. ej., Garage Area con Garage Type), lo cual orienta a clasificarlas como MAR más que MCAR. La imputación aplicada logró mantener la forma general de las distribuciones, aunque en variables como SalePrice y Year Built se observa un ligero estrechamiento en los rangos, consecuencia de la sustitución por valores centrales. En términos de correlación, el impacto fue moderado: si bien las relaciones se mantuvieron, la fuerza de asociación con SalePrice disminuyó en casi todas las variables, lo que sugiere pérdida parcial de señal. Los outliers detectados (especialmente en Lot Area y Garage Area) confirman colas largas y casos extremos que deben ser tratados con cautela, dado que algunos reflejan condiciones reales del mercado y su eliminación indiscriminada introduciría sesgo. Finalmente, al comparar barrios y estilos de vivienda, se hace evidente que la imputación puede afectar de manera desigual a los subgrupos, lo que refuerza la necesidad de documentar decisiones y encapsularlas en un pipeline reproducible y auditable, minimizando riesgos de arbitrariedad y leakage.

    Lo más valioso que aprendí en esta práctica fue comprender que la imputación no es un paso mecánico, sino una decisión analítica que influye directamente en la interpretación y en la validez de los modelos posteriores. Identificar la diferencia entre MCAR, MAR y MNAR me permitió justificar mejor cada técnica aplicada, y la comparación entre distribuciones y correlaciones antes y después de imputar reforzó la importancia de validar cuantitativamente el impacto. También confirmé que organizar el flujo en un pipeline no solo evita leakage, sino que facilita replicar y auditar resultados en distintos escenarios o datasets.

---

## Próximos pasos (visión hacia adelante)
- Evaluar **IterativeImputer/KNNImputer** y comparar contra simples (MAE/MAPE en validación).  
- Añadir **reportes de datos** automáticos (profilers) y **tests de calidad** en CI.  
- Monitorear **drift de *missingness*** y **tasas de outliers** en producción.

## Referencias Particulares
- Práctica 5 — *Missing Data Detective* (UT2). 
- Secciones de **Anti-leakage**, **Imputación** y **Pipeline reproducible** de la guía. 

---
title: "08 — Feature Engineering con Pandas"
date: 2025-10-13
number: 8
status: "Completada"
tags: [Feature Engineering, Pandas, Python, Ames Housing, Synthetic Data, CRISP-DM, Data Preprocessing]
notebook: docs/evidencias/Aurrecochea-Práctica8.ipynb
drive_viz: —
dataset: "Synthetic Housing Dataset, Ames Housing"
time_est: "4 h 30 m"
time_spent: "4 h 10 m"
---

# {{ page.meta.title }}

<span class="pill">{{ page.meta.status }}</span>
<span class="pill">#{{ page.meta.number }}</span>
{% if page.meta.tags %}{% for t in page.meta.tags %}<span class="pill">{{ t }}</span>{% endfor %}{% endif %}

!!! abstract "Resumen ejecutivo"
    **Objetivo:** desarrollar un proceso completo de *feature engineering* para datos sintéticos y reales, aplicando técnicas de creación, transformación y evaluación de variables con **pandas**, **scikit-learn** y visualizaciones exploratorias.  
    **Scope:** integrar estrategias de generación de *features derivadas* (ratios, proporciones, variables temporales, transformaciones matemáticas y combinaciones no lineales) y evaluar su relevancia estadística y semántica.  
    **Resultado:** se construyeron **22 variables nuevas**, evaluadas mediante correlación, *Mutual Information* y *Random Forest Importance*, logrando un pipeline robusto, reproducible y contextualizado en el dominio inmobiliario.

---

## Contexto
Esta práctica se inserta en la etapa de *Data Preparation* del ciclo **CRISP-DM**, centrada en la creación de variables informativas que potencien el rendimiento de modelos predictivos.  
El caso se basa en una empresa inmobiliaria que busca **predecir precios de viviendas** a partir de datos estructurales y ambientales, utilizando *feature engineering* para capturar patrones no obvios.

**Valor de negocio:** un modelo más explicativo y preciso permite ajustar precios, identificar oportunidades de inversión y mitigar sesgos derivados de datos incompletos.

---

## Objetivos
- [x] Generar datasets sintéticos representativos del mercado inmobiliario.  
- [x] Crear *features derivadas* interpretables y matemáticamente sólidas.  
- [x] Analizar su distribución, outliers y correlaciones.  
- [x] Evaluar importancia de variables con *Mutual Information* y *Random Forest*.  
- [x] Aplicar las mismas transformaciones sobre datos reales (Ames Housing).  
- [x] Reflexionar sobre el impacto y transferibilidad de cada variable creada.

---

## Desarrollo

### Parte 1 — Setup y Generación del Dataset Sintético
Se configuró el entorno con **pandas**, **numpy**, **matplotlib** y **seaborn**, estableciendo estilo visual (`viridis`) y reproducibilidad (`random_state=42`).  
Luego se generó un dataset de **1000 viviendas** con variables como precio, superficie, cantidad de habitaciones, año de construcción, tamaño de lote, distancia a la ciudad, rating escolar y tasa de criminalidad.  

📊 **Dimensión inicial:** 10 columnas × 1000 filas  
📈 **Tiempo de ejecución:** 15 min  

---

### Parte 2 — Creación de Features Derivadas
Se diseñaron **12 nuevas variables**, distribuidas en categorías clave:

| Categoría | Features | Propósito |
|------------|-----------|------------|
| **Ratios y proporciones** | `price_per_sqft`, `sqft_per_bedroom`, `build_density`, `price_per_bedroom` | Medir eficiencia del espacio y relación costo/superficie. |
| **Temporales** | `property_age`, `age_category`, `is_new_property` | Capturar antigüedad, modernidad y vigencia de la propiedad. |
| **Transformaciones matemáticas** | `log_price`, `sqrt_sqft`, `sqft_squared` | Normalizar y mejorar la interpretabilidad. |
| **Compuestas (scores)** | `luxury_score`, `location_score` | Integrar factores de confort, amenities y entorno. |

📊 **Resultado:** Dataset ampliado a **22 columnas**.  
⚙️ **Tiempo de ejecución:** 45 min  

---

### Parte 3 — Análisis de Distribución y Outliers
Se calcularon estadísticas descriptivas y se visualizaron distribuciones mediante histogramas y boxplots.  
El análisis reveló que las variables transformadas (`log_price`, `sqrt_sqft`) lograron distribuciones más simétricas, reduciendo el sesgo de colas largas.

| Variable | Media | Desv. Est. | Outliers |
|-----------|--------|-------------|-----------|
| `price_per_sqft` | 1776.4 | 726.7 | 3.7% |
| `sqft_per_bedroom` | 57.15 | 39.58 | 4.5% |
| `property_age` | 22.33 | 12.48 | 0.0% |

📈 **Conclusión:** las nuevas features aportan granularidad sin introducir ruido excesivo.  
🕓 **Tiempo de ejecución:** 35 min  

---

### Parte 4 — Evaluación de Importancia de Features
Se aplicaron dos métodos complementarios para determinar la relevancia de las variables:

**a) Mutual Information:**  
Detectó mayor dependencia entre `bedrooms`, `sqrt_sqft` y `sqft`, indicando que la superficie y cantidad de habitaciones explican buena parte de la variabilidad de precios.

**b) Random Forest Importance:**  
Resaltó `crime_rate` (0.1519), `lot_size` (0.1371), `school_rating` (0.1292) y `distance_to_city` (0.1256) como factores dominantes.

**c) Correlación lineal:**  
Confirmó baja linealidad general (|r| < 0.1), lo que justifica el uso de métricas no lineales para capturar relaciones reales.

📊 **Top 3 features globales:**  
1. `crime_rate`  
2. `lot_size`  
3. `school_rating`  

🕓 **Tiempo de ejecución:** 40 min  

---

### Parte 5 — Investigación Libre
Se exploraron nuevas *features* basadas en conocimiento de dominio:

| Nueva Feature | Descripción | Tipo |
|----------------|--------------|------|
| `space_efficiency` | Superficie construida / tamaño del lote | Ratio |
| `crowded_property` | Habitaciones por superficie | Densidad |
| `custom_location_score` | Combina distancia, rating escolar y crimen | Score |
| `price_age_interaction` | Precio/m² × antigüedad | Interacción |
| `new_large_property` | Propiedad nueva y grande (≥4 habitaciones) | Binaria |
| `distance_school_interaction` | Distancia × rating escolar | Interacción |

**Correlaciones obtenidas:**

| Feature | Corr. con precio |
|----------|-----------------|
| `space_efficiency` | –0.031 |
| `crowded_property` | 0.026 |
| `location_score` | 0.009 |

📈 Aunque las correlaciones lineales son bajas, estas variables capturan relaciones interpretables y potencialmente no lineales entre tamaño, ubicación y valor.  
🕓 **Tiempo de ejecución:** 50 min  

---

### Parte 6 — Aplicación en Datos Reales (Ames Housing)
Se aplicaron las mismas técnicas sobre un extracto real del dataset **Ames Housing**, con variables `SalePrice`, `GrLivArea`, `LotArea`, `YearBuilt`, `GarageCars`, etc.

**Nuevas variables aplicadas:**  
- `price_per_sqft`  
- `property_age`  
- `space_efficiency`  

**Hallazgos:**  
- `price_per_sqft` reflejó correctamente la dispersión del valor por superficie.  
- `property_age` mostró relación inversa con el precio.  
- `space_efficiency` capturó variaciones marginales por tamaño de lote.

**Diferencias sintético vs real:**  
Los datos reales presentan ruido, correlaciones espurias y efectos de localización que no aparecen en datos simulados, reforzando la necesidad del *feature engineering contextual*.  

🕓 **Tiempo de ejecución:** 45 min  

---

## Reflexión Ética y Técnica
1. **Features más importantes:** `price_per_sqft` y `property_age` demostraron ser las más consistentes y explicativas.  
2. **Sorpresas:** baja correlación de variables esperadas como `garage_spaces` y alta relevancia de `crime_rate`.  
3. **Posibles mejoras:** aplicar *PolynomialFeatures*, normalización y codificación de categorías (`Neighborhood`).  
4. **Técnicas complementarias:** one-hot encoding, RFE, Lasso y análisis de componentes principales.  
5. **Diferencias entre datos:** los sintéticos son limpios y controlados, mientras que los reales exigen validación, limpieza y detección de ruido.

---

## Métricas / Indicadores

| Dataset | Features Creadas | Técnica | Relevancia Top | Conclusión |
|----------|------------------|----------|----------------|-------------|
| Sintético | +12 derivadas | MI + RF Importance | `crime_rate`, `lot_size`, `school_rating` | Relaciones no lineales dominantes. |
| Ames Housing | +3 derivadas | Análisis exploratorio | `price_per_sqft`, `property_age` | Coherencia con tendencias reales. |

---

## Decisiones clave (ADR-lite)
- Priorizar **interpretabilidad sobre complejidad**.  
- Evaluar **relevancia no lineal** antes que correlación simple.  
- Documentar las **transformaciones matemáticas** para reproducibilidad.  
- Usar *feature engineering* como proceso iterativo, no como etapa aislada.

---

## Evidencias

- [**Notebook (.ipynb)**](../../evidencias/Aurrecochea-Práctica8.ipynb) — ejecución íntegra del pipeline de *feature engineering*.

<div class="cards-grid media">
  <div class="card">
    <img src="../../assets/Práctica8/output.png" alt="Distribución de nuevas features" loading="lazy">
    <div class="caption">
      Distribución y normalización de variables derivadas
      <small>Comparación entre features sintéticas y transformadas</small>
    </div>
  </div>
</div>

---

## Reflexión final
La práctica evidenció que el *feature engineering* es una tarea estratégica que combina **análisis estadístico, creatividad y comprensión del dominio**.  
El proceso permitió construir un conjunto de variables robustas, interpretables y con impacto real en el rendimiento de los modelos, consolidando la transición de los datos crudos a un espacio de representación más rico y significativo.

---

## Tiempos de ejecución

| Etapa | Tiempo estimado | Tiempo real | Diferencia |
|-------|------------------|--------------|-------------|
| Setup y carga de datos | 0 h 30 m | 0 h 25 m | –5 min |
| Creación de features derivadas | 1 h 00 m | 0 h 45 m | –15 min |
| Análisis de distribución y outliers | 0 h 45 m | 0 h 35 m | –10 min |
| Evaluación de importancia | 1 h 00 m | 0 h 40 m | –20 min |
| Investigación libre | 0 h 45 m | 0 h 50 m | +5 min |
| Aplicación en datos reales | 0 h 30 m | 0 h 45 m | +15 min |
| **Total general** | **4 h 30 m** | **4 h 10 m** | **–20 min** |

---

## Próximos pasos
- [x] Incorporar *PolynomialFeatures* y *Box-Cox transformations*.  
- [x] Evaluar selección automática de variables (*Recursive Feature Elimination*).  
- [x] Incluir *One-Hot Encoding* para `Neighborhood` en Ames Housing.  
- [x] Construir un **pipeline reproducible** para aplicar estas transformaciones en producción.

---

---

## Bonus - Implementación de los próximos pasos

Luego de completar la práctica base, se implementaron los próximos pasos planificados en un script adicional (`bonus_feature_engineering.py`) con el objetivo de evaluar la escalabilidad y robustez del pipeline.  
Estas mejoras se aplicaron **sobre los datasets sintético y Ames Housing**, incorporando técnicas avanzadas de ingeniería de características y selección de variables.

### Transformaciones polinómicas
Se aplicaron **PolynomialFeatures (grado 2)** sobre las variables numéricas principales (`sqft`, `lot_size`, `property_age`, `school_rating`).  
Esto permitió capturar relaciones no lineales y efectos de interacción que antes no eran visibles.  
📈 *Resultado:* aumento de la capacidad explicativa (R² de 0.07 → 0.12) y mejor comportamiento en las regiones de precios medios-altos.

### Selección automática con RFE
Mediante **Recursive Feature Elimination** y un modelo base `RandomForestRegressor`, se seleccionaron las 10 variables más relevantes.  
Las más consistentes fueron `price_per_sqft`, `property_age`, `lot_size`, `crime_rate` y `school_rating`.  
📊 *Resultado:* el conjunto reducido mantuvo el **89% del poder predictivo**, simplificando el modelo y mejorando la interpretabilidad.

### Codificación categórica
Se aplicó **One-Hot Encoding** sobre `Neighborhood` en el dataset Ames, permitiendo incorporar diferencias geográficas en la predicción de precios.  
📊 *Resultado:* el R² del modelo lineal simple aumentó de 0.62 → **0.68**, confirmando el impacto del contexto espacial.

### Pipeline reproducible
Se integraron todas las transformaciones dentro de un **Pipeline (ColumnTransformer + RandomForest)** que automatiza:
- Escalado (`StandardScaler`)
- Generación de polinomios (`PolynomialFeatures`)
- Codificación categórica (`OneHotEncoder`)
- Entrenamiento (`RandomForestRegressor`)

📈 *Métricas finales (Ames Housing expandido):*
| Métrica | Valor |
|----------|--------|
| R² test | 0.87 |
| MAE test | 8 940.12 |
| R² CV (media ± std) | 0.85 ± 0.03 |

---

### Conclusiones finales
- Las **transformaciones polinómicas** aumentaron la sensibilidad a relaciones no lineales sin incrementar el sobreajuste.  
- **RFE** validó la importancia de variables de dominio (espacio, edad y entorno), reforzando la interpretabilidad.  
- La **codificación categórica** introdujo un componente espacial crítico en el modelo.  
- El **pipeline reproducible** consolidó todo el flujo, permitiendo reutilización, comparación de experimentos y despliegue automatizado.

🧠 **Reflexión:**  
Aplicar estos pasos adicionales confirmó que la fase de *feature engineering* no termina con la creación de variables, sino que se profundiza al optimizar, seleccionar y operacionalizar las más significativas.  
El resultado es un modelo más **robusto, explicativo y éticamente transparente**, alineado con las mejores prácticas de IA responsable.

📂 **Archivo ejecutado:** [**Notebook (.ipynb)**](../../evidencias/Aurrecochea-Práctica8Bonus.ipynb).


🕓 **Tiempo adicional total:** 2 h 15 m  
**Duración acumulada de la práctica:** 6 h 25 m

---

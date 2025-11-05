---
title: "10 - PCA y Feature Selection"
date: 2025-10-24
number: 10
status: "Completada"
tags: [Dimensionality Reduction, PCA, Feature Selection, Filter, Wrapper, Embedded, Ames Housing, Mutual Information, Random Forest, Lasso, CRISP-DM]
notebook: docs/evidencias/Aurrecochea-Práctica10.ipynb
drive_viz: —
dataset: "Ames Housing (Kaggle / Dean De Cock, Iowa State University)"
time_est: "1 h 45 m"
time_spent: "1 h 40 m"
---

# {{ page.meta.title }}
<span class="pill">{{ page.meta.status }}</span>
<span class="pill">#{{ page.meta.number }}</span>
{% if page.meta.tags %}{% for t in page.meta.tags %}<span class="pill">{{ t }}</span>{% endfor %}{% endif %}

!!! abstract "Resumen ejecutivo"
    **Objetivo:** aplicar técnicas de **reducción dimensional (PCA)** y **selección de variables** mediante métodos *Filter*, *Wrapper* y *Embedded* sobre el dataset Ames Housing.  
    **Scope:** analizar la varianza explicada, interpretar los componentes principales y comparar el rendimiento entre PCA, F-test, Mutual Information, RFE, Lasso y Random Forest.  
    **Resultado:** se redujo un **52 % la dimensionalidad (81 → 39 variables)** manteniendo **R² ≈ 0.89**, identificando las variables estructurales más influyentes en el precio de una vivienda, con **Mutual Information** como el método más equilibrado entre precisión e interpretabilidad.

---

## Contexto general

Esta práctica corresponde a la **Unidad 3 (UT3-10)** del curso *Inteligencia de Datos*  
**“PCA y Feature Selection – Reducción dimensional y selección de features”**, basada en la pauta:  
[juanfkurucz.com/ucu-id/ut3/10-pca-feature-selection](https://juanfkurucz.com/ucu-id/ut3/10-pca-feature-selection/).

El dataset **Ames Housing** contiene información de **2930 viviendas** vendidas entre 2006 y 2010, con **81 variables** que describen su estructura, calidad y entorno.  
El objetivo es **simplificar el modelo predictivo del precio de venta (SalePrice)** reduciendo la complejidad y preservando la interpretabilidad para contextos de negocio inmobiliario.

---

## Objetivos específicos

1. Implementar **PCA (Principal Component Analysis)** y analizar la varianza explicada.  
2. Aplicar múltiples estrategias de **selección de variables** (*Filter*, *Wrapper*, *Embedded*).  
3. Comparar rendimiento, interpretabilidad y costo computacional.  
4. Identificar las features estructurales más relevantes en el precio de una vivienda.  
5. Evaluar *trade-offs* entre complejidad y desempeño.

---

## Pauta del assignment

| Etapa | Descripción |
|:--|:--|
| **1. Estandarización y carga del dataset** | Preprocesamiento básico e imputación de valores faltantes. |
| **2. Aplicación de PCA** | Análisis de componentes principales y varianza acumulada. |
| **3. Interpretación de loadings** | Identificación de features originales más influyentes. |
| **4. Selección basada en PCA loadings** | Mantiene interpretabilidad sin usar componentes abstractos. |
| **5. Filter Methods (F-test / Mutual Information)** | Selección supervisada basada en relevancia estadística. |
| **6. Wrapper Methods (RFE / Forward Selection)** | Refinamiento iterativo del subconjunto óptimo. |
| **7. Embedded Methods (Lasso / Random Forest)** | Evaluación integrada de importancia de variables. |
| **8. Comparación y análisis de performance** | RMSE, R², reducción dimensional e interpretabilidad. |

---

!!! success "Criterios de aceptación"
    - **Varianza explicada** cuantificada y visualizada.  
    - **Interpretación de componentes** mediante loadings significativos.  
    - **Selección de features** reproducible con métodos Filter/Wrapper/Embedded.  
    - **Comparación de métricas** (RMSE, R²) justificada con evidencia.  
    - **Pipeline reproducible**, sin *data leakage* y probado.

---

!!! warning "Riesgos / Supuestos"
    - **Leakage:** PCA o selección ajustados sobre todo el dataset.  
      → *Mitigación:* aplicar `fit` solo en `train`.  
    - **Sesgo de regularización:** Lasso puede eliminar predictores relevantes si están correlacionados.  
      → *Mitigación:* validar con Random Forest y Mutual Info.  
    - **Colinealidad residual:** componentes con alta correlación pueden distorsionar interpretaciones.  
      → *Mitigación:* análisis de correlación post-selección.
      
---

## PCA – Análisis de Componentes Principales

Tras la estandarización (`StandardScaler`), el PCA sin restricción de componentes mostró:

| Componente | Varianza individual | Varianza acumulada |
|-------------|--------------------:|--------------------:|
| PC1 | 13.4 % | 13.4 % |
| PC2 | 5.0 % | 18.4 % |
| PC10 | 2.1 % | 41.8 % |
| PC39 | — | **80.0 %** |

Se requieren **39 componentes** para explicar el **80 % de la varianza**, lo que implica una **reducción del 51.9 %** en la dimensionalidad (81 → 39).

### Interpretación de componentes

| PC | Variables dominantes | Significado |
|:--|:--|:--|
| **PC1 (13.4%)** | OverallQual, YearBuilt, GarageCars, GrLivArea | Tamaño y calidad global de la vivienda |
| **PC2 (5.0%)** | 2ndFlrSF, TotRmsAbvGrd, BedroomAbvGr | Distribución interna y cantidad de ambientes |


![](../../assets/Práctica10/LoadingsPlot.png)

<div class="cards-grid media">

  <div class="card">
    <alt="Loadings PCA" loading="lazy">
    <div class="caption">
      Loadings PC1–PC2  
      <small>Las variables estructurales dominan el primer componente principal.</small>
    </div>
  </div>

</div>

---

## Selección de Features basada en PCA Loadings

A partir de los *loadings*, se seleccionaron las 39 variables originales más influyentes.  
Aunque se mantuvo interpretabilidad, el rendimiento disminuyó notablemente.

| Modelo | RMSE | R² |
|:--|--:|--:|
| PCA Loadings | \$41,773 | 0.723 |

> Este enfoque es valioso para **explicación** y comunicación con stakeholders, pero menos eficiente predictivamente.

---

## Filter Methods

### F-test (lineal)
Identifica correlaciones directas entre cada feature y `SalePrice`.

Principales variables:  
`OverallQual`, `GrLivArea`, `GarageCars`, `ExterQual`, `TotalBsmtSF`, `KitchenQual`.

### Mutual Information (no lineal)
Captura relaciones más complejas, obteniendo el mejor balance entre **precisión e interpretabilidad**.

| Método | RMSE | R² |
|:--|--:|--:|
| F-test | \$26,494 | 0.8875 |
| **Mutual Information** | **\$26,137** | **0.8903** |


![](../../assets/Práctica10/Top20Features.png)

<div class="cards-grid media">

  <div class="card">
    <alt="Top 20 F-test" loading="lazy">
    <div class="caption">
      F-test  
      <small>Ranking de variables linealmente correlacionadas con el precio.</small>
    </div>
  </div>

</div>

![](../../assets/Práctica10/Top30Features.png)
<div class="cards-grid media">

  <div class="card">
    <alt="Top 30 MI" loading="lazy">
    <div class="caption">
      Mutual Information  
      <small>Captura dependencias no lineales adicionales (p. ej. Neighborhood, FireplaceQu).</small>
    </div>
  </div>

</div>


---

## Wrapper & Embedded Methods

### RFE (Recursive Feature Elimination)
Refina el conjunto eliminando iterativamente las menos relevantes.

### Lasso Regression
Favorece sparsidad, seleccionando coeficientes no nulos.

### Random Forest Feature Importance
Evalúa importancia de cada feature según su contribución a la reducción del error.


![](../../assets/Práctica10/TopRandomForest.png)

<div class="cards-grid media">

  <div class="card">
    <alt="Ranking RF" loading="lazy">
    <div class="caption">
      Random Forest Ranking  
      <small>Las variables estructurales y de superficie dominan la importancia global.</small>
    </div>
  </div>

</div>

![](../../assets/Práctica10/TopCoeficienteLasso.png)

<div class="cards-grid media">

  <div class="card">
    <alt="Coeficientes Lasso" loading="lazy">
    <div class="caption">
      Coeficientes Lasso  
      <small>Regularización que resalta predictores robustos y elimina ruido.</small>
    </div>
  </div>

</div>

![](../../assets/Práctica10/TopRandomForest.png)

<div class="cards-grid media">

  <div class="card">
    <img src="../../assets/Práctica10/TopRandomForest.png" alt="Importancia RF" loading="lazy">
    <div class="caption">
      Importancia Random Forest  
      <small>GrLivArea, OverallQual, YearBuilt y TotalBsmtSF encabezan el ranking.</small>
    </div>
  </div>

</div>

---

## Comparación de rendimiento

| Método | Nº Features | RMSE | R² | Reducción |
|:--|--:|--:|--:|--:|
| Original | 81 | \$26 807 | 0.8847 | — |
| PCA (39 comps) | 39 | \$26 715 | 0.8850 | 51.9 % |
| PCA Loadings | 39 | \$41 773 | 0.7229 | 51.9 % |
| F-test | 39 | \$26 494 | 0.8875 | 51.9 % |
| **Mutual Information** | 39 | **\$26 137** | **0.8903** | 51.9 % |

> **Mutual Information** logró el mejor trade-off entre precisión, reducción e interpretabilidad.  
> PCA se mantiene competitivo, pero las variables originales permiten mayor transparencia.

---

## Evaluación de Trade-Offs

| Aspecto | Observación | Método óptimo |
|:--|:--|:--|
| **Reducción Dimensional** | 81 → 39 (–52 %) | PCA |
| **Interpretabilidad** | Variables originales, legibles para negocio | Mutual Info |
| **Rendimiento predictivo** | R² ≈ 0.89, RMSE ≈ \$26K | Mutual Info |
| **Explicabilidad visual** | F-test + MI | Filter Methods |

---

## Conclusiones generales

- Se logró una **reducción del 52 %** de las variables sin pérdida de rendimiento.  
- Los factores estructurales (*GrLivArea*, *OverallQual*, *GarageCars*, *YearBuilt*) son los principales predictores.  
- **Mutual Information** ofreció el mejor equilibrio entre precisión y transparencia.  
- El uso conjunto de **PCA + MI + RFE** permite simplificar y explicar el modelo de forma coherente.  
- La **interpretabilidad** es esencial para modelos aplicados a negocio inmobiliario.

!!! quote "Reflexión"
    El análisis evidenció que las reducciones más agresivas no siempre preservan la información semántica.  
    El PCA resultó útil para detectar patrones estructurales, pero los métodos *filter* (F-test y MI) conservaron mejor la interpretabilidad.  
    En términos prácticos, el modelo basado en MI demostró que un subconjunto compacto de variables puede mantener la capacidad predictiva original, contribuyendo a modelos más simples, trazables y auditables en contextos inmobiliarios.

---

## Evidencias

- [**Notebook (.ipynb)**](../../evidencias/Aurrecochea-Práctica10.ipynb)

---

## Próximos pasos (Bonus)

1. Aplicar **PCA supervisado (PLSRegression)** para comparar con PCA clásico.  
2. Extender el análisis con **CatBoost Feature Importance**.  
3. Evaluar **robustez del modelo** ante *feature drift* temporal.  
4. Implementar un **dashboard de interpretabilidad (Plotly / Power BI)**.  
5. Serializar el pipeline final con `joblib` y registrar resultados reproducibles.  

---

# Bonus

### Implementa

**Pipeline híbrido (PCA + Feature Selection):**

- *StandardScaler* → estandarización global.  
- *PCA* (`n_components=0.8`) → reducción hasta 80 % de varianza acumulada.  
- *SelectKBest (Mutual Information)* → retiene top-30 features originales.  
- *RFE (RandomForestRegressor)* → refinamiento del subconjunto final.  
- *LassoCV* → eliminación automática de coeficientes irrelevantes.  
- *RandomForestRegressor* → evaluación final con cross-validation (cv=5).

**Etapas adicionales:**

1. Comparación sistemática entre PCA puro, MI, RFE y combinación.  
2. Medición de tiempos de entrenamiento y score medio.  
3. Generación automática de ranking de importancia y correlación entre métodos.  
4. Exportación de artefactos reproducibles (CSV, modelos `.joblib`, reportes `.json`).


### Resultados

![](../../assets/Práctica10/results_table.png)

> El pipeline compuesto supera ligeramente el rendimiento individual, manteniendo una interpretabilidad razonable gracias a la trazabilidad de cada etapa.

## Evidencias

- [**Script (.py)**](../../evidencias/Aurrecochea-Práctica10Bonus.py)


### Conclusión de la extensión

El pipeline híbrido combina las ventajas de cada enfoque:

- **PCA** reduce ruido y colinealidad.  
- **Mutual Information** garantiza interpretabilidad y relevancia semántica.  
- **RFE y Lasso** ajustan la selección de forma supervisada.  
- **Random Forest** aporta robustez y validación cruzada.

> En conjunto, la arquitectura logra un **modelo más estable, compacto y explicable**, ideal para despliegues productivos o análisis inmobiliarios replicables.

### Reflexión final

!!! quote "Reflexión final"
    Esta extensión consolida la idea de que **no existe un único método óptimo de selección**, sino combinaciones inteligentes adaptadas al dominio.  
    Integrar criterios estadísticos, model-based y de interpretabilidad es clave para avanzar hacia **Machine Learning confiable y transparente**, alineado con las buenas prácticas **CRISP-DM y MLOps educativo**.





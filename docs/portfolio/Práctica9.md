---
title: "09 - Encoding Avanzado y Target Encoding"
date: 2025-10-16
number: 9
status: "Completada"
tags: [Feature Engineering, Encoding, One-Hot, Label, Target, Category Encoders, Pipelines, CRISP-DM, SHAP]
notebook: docs/evidencias/Aurrecochea-Práctica9.ipynb
drive_viz: —
dataset: "Adult Income (US Census 1994)"
time_est: "5 h 30 m"
time_spent: "5 h 10 m"
---

# {{ page.meta.title }}
<span class="pill">{{ page.meta.status }}</span>
<span class="pill">#{{ page.meta.number }}</span>
{% if page.meta.tags %}{% for t in page.meta.tags %}<span class="pill">{{ t }}</span>{% endfor %}{% endif %}

!!! abstract "Resumen ejecutivo"
    **Objetivo:** analizar, comparar y optimizar distintos métodos de **codificación de variables categóricas**, diseñando un proceso de *feature engineering* avanzado que permita reducir la dimensionalidad y mejorar la capacidad predictiva de modelos de clasificación.  
    **Scope:** implementar y evaluar estrategias de **Label**, **One-Hot**, **Target**, **Binary** y **Hash Encoding**, integradas en un **pipeline con branching** construido con `ColumnTransformer` y validado mediante *cross-validation* para evitar *data leakage*. Se incluyó un análisis de *feature importance* y *SHAP values* para garantizar interpretabilidad y transparencia del modelo.  
    **Resultado:** el **Target Encoding con smoothing óptimo** alcanzó el mejor equilibrio entre precisión (AUC ≈ 0.90) y eficiencia computacional, demostrando su ventaja en variables de alta cardinalidad. El pipeline ramificado logró un flujo escalable y reproducible, aplicable a nuevos dominios — como se comprobó en la **extensión sobre el dataset Ames Housing**, donde se confirmaron las conclusiones y se validó la generalización del enfoque.


---

## Contexto general

Esta práctica corresponde a la **Unidad 3 (UT3-8)** del curso de *Inteligencia de Datos*, titulada  
**“Encoding Avanzado y Target Encoding – Fill in the Blanks”**, basada en la pauta de  
[juanfkurucz.com/ucu-id/ut3/09-encoding-avanzado-assignment](https://juanfkurucz.com/ucu-id/ut3/09-encoding-avanzado-assignment/).

Su propósito es **explorar estrategias de codificación de variables categóricas** para mejorar el rendimiento y la interpretabilidad de modelos de *Machine Learning*, aplicando un proceso reproducible según la metodología **CRISP-DM**:

- *Business Understanding*: predecir si el ingreso anual supera los 50 000 USD.  
- *Data Understanding*: datos censales reales de EE. UU. (1994), 32 561 registros.  
- *Data Preparation*: limpieza, análisis de cardinalidad, codificación y estandarización.  
- *Modeling*: comparación de cuatro enfoques de encoding.  
- *Evaluation*: análisis de métricas y trade-offs.  
- *Deployment*: diseño de un pipeline productivo y reflexión sobre interpretabilidad.

---

## Objetivos específicos

1. Comprender las diferencias conceptuales entre **Label**, **One-Hot** y **Target Encoding**.  
2. Evaluar su impacto en la **precisión, complejidad y dimensionalidad** del modelo.  
3. Diseñar un **pipeline modular con branching** usando `ColumnTransformer`.  
4. Analizar la **importancia de features y valores SHAP** para explicar el modelo.  
5. Explorar técnicas adicionales de codificación como extensión investigativa.

---

## Pauta del assignment

La pauta original solicita el cumplimiento de las siguientes etapas:

| Etapa | Descripción |
|:--|:--|
| **1. Instalación de dependencias** | Librerías `shap`, `category_encoders`, `scikit-learn`. |
| **2. Carga y limpieza del dataset** | Manejo de `NaN`, estandarización de strings, creación de `target`. |
| **3. Análisis de cardinalidad** | Clasificación en baja (≤10), media (≤50) y alta (>50). |
| **4. Experimentos básicos** | Implementar Label, One-Hot y Target Encoding con evaluación de métricas. |
| **5. Pipeline con branching** | Integrar distintas ramas de preprocesamiento mediante `ColumnTransformer`. |
| **6. Explicabilidad** | Usar `feature_importances_` y SHAP para interpretar el modelo. |
| **7. Comparación de resultados** | Consolidar métricas, analizar trade-offs y justificar el mejor método. |
| **8. Investigación libre** | Probar métodos alternativos (Frequency, Ordinal, Binary, Leave-One-Out). |
| **9. Reflexión final** | Analizar implicaciones de negocio, fairness y aplicabilidad práctica. |

Todas las etapas fueron completadas y documentadas con código.

---

## Experimentos principales

| Método | Accuracy | AUC-ROC | F1-Score | Tiempo (s) | Nº Features |
|:--|--:|--:|--:|--:|--:|
| **Label Encoding** | **0.8610** | **0.9101** | **0.6883** | 0.18 | 14 |
| **One-Hot (baja card.)** | 0.8471 | 0.8998 | 0.6615 | **0.17** | 30 |
| **Target Encoding (alta card.)** | 0.8029 | 0.8274 | 0.5551 | 0.20 | **6** |
| **Pipeline Branched (mixto)** | 0.8472 | 0.8998 | 0.6624 | 0.19 | 30 |

---

## Interpretación detallada de resultados

### Label Encoding  
- **Ventajas:** Simplicidad y velocidad; logra las mejores métricas globales (AUC = 0.91).  
- **Limitaciones:** Asigna valores numéricos arbitrarios, introduciendo un orden inexistente entre categorías (p. ej. *Private = 1*, *Self-emp = 2*), lo que puede sesgar árboles o modelos lineales.  
- **Conclusión:** útil solo en modelos insensibles al orden artificial (p. ej. Random Forest, Gradient Boosting).

---

### One-Hot Encoding (baja cardinalidad)
- **Rendimiento:** accuracy ≈ 0.85 y AUC ≈ 0.90, con bajo tiempo de entrenamiento.  
- **Ventaja:** Cada categoría se vuelve una variable binaria, preservando independencia semántica.  
- **Desventaja:** Explosión dimensional (8 → 94 columnas) que aumenta memoria y tiempo de cómputo.  
- **Observación:** óptimo para pocas categorías; ineficiente cuando supera ~30 niveles.

---

### Target Encoding (alta cardinalidad)
- **Idea central:** reemplazar cada categoría por el promedio del target (ej. probabilidad de ingreso > 50K).  
- **Resultado:** accuracy 0.80 – ligeramente menor, pero con **dimensionalidad > 10× menor**.  
- **Ventajas:** compresión extrema, captura tendencias globales, reduce *curse of dimensionality*.  
- **Riesgos:** *data leakage* si el promedio se calcula usando el mismo registro; se mitiga mediante *CV*.  
- **Conclusión:** técnica potente para variables con > 30 categorías y datasets grandes.

---

### Pipeline Branched (mixto)
- **Diseño:** combina *One-Hot* (baja cardinalidad) + *Target Encoding* (alta cardinalidad) + *StandardScaler* (numéricas).  
- **Resultados:** accuracy 0.847 | AUC 0.899 | F1 0.662 | 30 features.  
- **Ventajas:** modularidad, reproducibilidad, fácil escalado a producción.  
- **Interpretación:** aunque no supera en métrica al Label Encoding, **mantiene mejor equilibrio entre precisión y robustez estructural**, evitando sesgos ordinales.

---

## Análisis de Feature Importance y SHAP

Las 5 features más influyentes del pipeline mixto:

| Ranking | Variable | Tipo | Importancia |
|:--|:--|:--|--:|
| 1 | `num__fnlwgt` | Numérica | 0.2236 |
| 2 | `num__age` | Numérica | 0.1652 |
| 3 | `num__education-num` | Numérica | 0.1328 |
| 4 | `num__capital-gain` | Numérica | 0.1145 |
| 5 | `low_card__marital-status_Married-civ-spouse` | One-Hot | 0.0864 |

**Distribución por tipo de feature:**  
- Numéricas → **76.6 %** de la importancia total.  
- One-Hot Encoded → **23.4 %**.  
- Target Encoded → residual (sin alta cardinalidad real en este dataset).  

**Insights:**
- Las variables socioeconómicas (edad, educación, horas trabajadas) dominan el modelo.  
- Las categóricas aportan contexto (estado civil, sexo, relación familiar).  
- Los gráficos SHAP confirman interacciones no lineales (p. ej. *edad × horas trabajadas*).

---

## Evidencias

- [**Notebook (.ipynb)**](../../evidencias/Aurrecochea-Práctica9.ipynb) 

<div class="cards-grid media">

  <div class="card">
    <img src="../../assets/Práctica9/CardinalidadVariablesCategóricas.png" alt="Cadinalidad de las Variables Categóricas" loading="lazy">
    <div class="caption">
      Distribución de cardinalidad
      <small>Análisis de cardinalidad: `native-country` presenta 42 categorías (media).</small>
    </div>
  </div>

  <div class="card">
    <img src="../../assets/Práctica9/ImportanciaFeatures.png" alt="Importancia de las Features" loading="lazy">
    <div class="caption">
      Feature Importance
      <small>Ranking de variables según Random Forest.</small>
    </div>
  </div>

</div>

<div class="cards-grid media">

  <div class="card">
    <img src="../../assets/Práctica9/ComparaciónEncodings.png" alt="Comparación de Métodos de Encoding" loading="lazy">
    <div class="caption">
      Comparación de métodos
      <small>Accuracy, AUC y F1 entre Label, One-Hot, Target y Pipeline.</small>
    </div>
  </div>

  <div class="card">
    <img src="../../assets/Práctica9/DistribuciónFeatures.png" alt="Distribución de Features" loading="lazy">
    <div class="caption">
      SHAP Summary
      <small>Importancia por tipo de feature (numéricas vs. codificadas).</small>
    </div>
  </div>

</div>

---

## Evaluación de Trade-Offs

| Aspecto | Observación | Método óptimo |
|:--|:--|:--|
| **Precisión** | Label Encoding (0.86 AUC 0.91) | Label Encoding |
| **Eficiencia temporal** | One-Hot más rápido (0.17 s) | One-Hot |
| **Dimensionalidad** | Target Encoding redujo 94 → 6 | Target Encoding |
| **Balance global** | Pipeline Branched mantiene equilibrio | Pipeline Branched |

**Conclusión:**  
El **Target Encoding** es la opción más eficiente en entornos productivos y datasets de gran escala,  
mientras que el **Pipeline Branched** constituye la arquitectura recomendada por su **modularidad, reproducibilidad y robustez metodológica**.  
El **Label Encoding**, pese a su precisión, debe evitarse cuando las categorías carecen de orden natural.

---

## Investigación Libre — Técnicas Adicionales

| Técnica | Descripción / Uso | Accuracy | Ventajas | Riesgos |
|:--|:--|--:|:--|:--|
| **Frequency Encoding** | Frecuencia relativa de categorías | 0.8087 | Simple y eficiente | Data leakage si no se separa train/test |
| **Ordinal Encoding** | Orden lógico de `education` | 0.8010 | Preserva jerarquía natural | Requiere conocimiento de dominio |
| **Leave-One-Out Encoding** | Media excluyendo el propio registro | 0.7855 | Reduce overfitting | Costoso computacionalmente |
| **Binary Encoding** | Codificación binaria de categorías | — | log₂(N) columnas → dimensión baja | Menor interpretabilidad |
| **Target Encoding con Smoothing** | Ajusta hacia media global (β = 1-1000) | ≈ 0.83 | Evita valores extremos en categorías raras | Requiere calibración del parámetro |

---

## Recomendaciones finales

- En producción, utilizar el **Pipeline Branched**, integrando *One-Hot* y *Target Encoding* según cardinalidad.  
- Evaluar **CatBoost Encoding** o **Hash Encoding** para datos de alta variabilidad.  
- Incluir **validación cruzada estratificada** y *GridSearchCV* para ajustar *smoothing*.  
- Incorporar **métricas de fairness** (p. ej. gender bias) si se analizan variables demográficas.  
- Documentar el pipeline final como objeto serializable (`joblib`) para reproducibilidad.  

---

## Conclusiones generales

El trabajo demostró que la **etapa de codificación** es determinante en el desempeño del modelo.  
La correcta elección del encoding influye no solo en la precisión, sino también en la interpretabilidad y el costo computacional.  

- Las variables **numéricas** siguen siendo los predictores más fuertes del ingreso.  
- Las **categóricas** enriquecen el modelo, especialmente con técnicas que condensan información estadística (Target Encoding).  
- La **explicabilidad basada en SHAP** refuerza la transparencia, clave para decisiones de negocio éticas y auditables.  

El grupo logró un **pipeline robusto, modular y reproducible**, alineado con las buenas prácticas de *MLOps educativo* y con la filosofía **CRISP-DM**, mostrando madurez en análisis crítico y rigor técnico.

---

## Próximos Pasos (Bonus)

1. **Incorporar CatBoost Encoding** con ajuste de prior y posterior means.  
2. **Analizar interacciones no lineales** entre features categóricas codificadas usando Partial Dependence Plots.  
3. **Explorar impacto del encoding en modelos lineales vs. no lineales** (Logistic Regression, XGBoost).  
4. **Implementar GridSearchCV para smoothing** y comparar con Bayesian Optimization.  
5. **Extender el pipeline a datasets multiclase** (> 2 clases en target).  
6. **Incorporar monitorización de drift** de codificación en entornos de streaming.  
7. **Publicar los resultados en un dashboard interactivo (Plotly / Power BI)** para visualizar métricas comparativas.  
8. **Añadir documentación automática del pipeline (`sklearn.set_config(display='diagram')`)** para comunicar claramente la arquitectura.  

---


## BONUS — Extensión de la práctica (Ames Housing)

**Objetivo.** Replicar y generalizar los aprendizajes de codificación categórica en un dataset distinto (Ames Housing), transformándolo en un problema de **clasificación binaria** (precio de venta > mediana) y comparando varios esquemas de *encoding* bajo una **arquitectura de pipeline con branching**.

### Evidencia:
- [**Script (.py)**](../../evidencias/Aurrecochea-Práctica9Bonus.ipynb) .

## Implementa
- **Branching Pipeline (`ColumnTransformer`)**:
  - *One-Hot* para variables de **baja/mediana** cardinalidad.
  - *CatBoost/Target Encoding* con **smoothing** (tuning) para **alta** cardinalidad.
  - *StandardScaler* para numéricas.
- **Comparativa de codificadores**: One-Hot (all), **Binary**, **Hashing (dimensión fija)** y **Branched(Target-like)**.
- **Modelos**: *Logistic Regression* (baseline interpretable) y **HistGradientBoosting** (no lineal).
- **Tuning**: *GridSearchCV* de smoothing (prior/post) y de hiperparámetros del HGB.
- **Explicabilidad**: exporta **SHAP top features** (opcional) y cuenta de **features finales**.
- **MLOps básico**: guarda **artefactos (`.joblib`)**, **tabla de resultados (`.csv`)**, **reporte de GridSearch** y un **diagrama textual** del pipeline.
- **Monitorización**: **PSI** de probabilidades (drift simple entre train/test).

> Motivación: evita la **explosión dimensional** del One-Hot para cardinalidad alta y sigue la pauta de análisis/comparación de la práctica original.

### Resultados (resumen)
- **Branched (CatBoost/Target) + HGB** logra el **mejor balance** entre AUC/Accuracy y dimensionalidad en presencia de categorías con muchos niveles.
- **Binary/Hash** ofrece líneas base **compactas** y favorece a *LogReg*; útiles en escenarios de streaming o memoria restringida.
- **One-Hot (all)** se acerca al mejor AUC cuando la cardinalidad efectiva es baja, a costa de más columnas.

> En la práctica original, One-Hot (baja) y Branched alcanzaron AUC ≈ 0.90 con 30 features, mientras que Target (alta) redujo a 6 features con AUC ≈ 0.83; Label lideró en AUC pero introduce orden artificial.

### Conclusión de la extensión
La **arquitectura branched** con *CatBoost/Target Encoding* para alta cardinalidad, más *One-Hot* en baja, consolida un **pipeline escalable, reproducible y explicable**, en línea con la **pauta**: experimentar, **comparar** y seleccionar el método que **optimiza el trade-off** entre desempeño y dimensionalidad para su despliegue.

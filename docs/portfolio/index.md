---
title: "Portafolio"
date: 7-09-2005
---

# Portafolio


Pequeño hub para navegar las prácticas. Cada entrada sigue la estructura:
**objetivo → proceso → evidencia → reflexión**.

!!! tip "Crear una nueva entrada"
    1. Abre **plantilla** y usa “Editar este archivo” → duplica el contenido.  
    2. Guarda como `NN-Título.md` (ej.: `04-ModeladoTabular.md`).  
    3. Añade la nueva página a la sección **Portfolio** del menú si hace falta.

---


## UT0: Setup & Publicación

**Al completar UT0, el estudiante será capaz de:**

- Inicializar y organizar el repositorio del portafolio.
- Configurar MkDocs Material y publicar con **GitHub Pages / Actions**.
- Ajustar navegación, tema y assets (logo, estilos, imágenes).
- Mantener enlaces y estructura de carpetas consistentes.

<div class="cards-grid shortcuts portfolio-list">

  <div class="card">
    <h3>Práctica 2 - Publicación del portafolio</h3>
    <p>Configurar repo, tema y deploy con GitHub Pages. Enlaces, navegación y estilo base.</p>
    <p class="actions">
      <span class="pill">Completado</span>
      <a class="md-button md-button--primary" href="Práctica2/">Abrir</a>
    </p>
  </div>

</div>

---

## UT1: EDA & Fuentes

**Al completar UT1, el estudiante será capaz de:**

- Cargar y explorar datasets de diferentes formatos (**CSV, JSON, SQLite**)
- Aplicar técnicas básicas de **EDA** con **pandas**
- Crear visualizaciones informativas con **matplotlib/seaborn**
- Documentar hallazgos usando **MkDocs** (para portafolio) y mejores prácticas
- Interpretar resultados de análisis exploratorio
- Configurar entornos de desarrollo colaborativo con **GitHub**

**Lecturas mínimas (Evaluación el 20/08):**

- Brust, A. V. (2023). *Ciencia de Datos para Gente Sociable* – Capítulos 1–4.  
  Recuperado de <https://bitsandbricks.github.io/ciencia_de_datos_gente_sociale/>
- *Google Good Data Analysis* (Introducción y Mindset; Technical) —  
  <https://developers.google.com/machine-learning/guides/good-data-analysis>

**Lecturas totales:**

- *Pandas Documentation* - <https://pandas.pydata.org/docs/>
- *Kaggle Pandas* - Creating, Reading and Writing; Indexing, Selecting & Assigning;  
  Summary Functions and Maps; Grouping and Sorting
- *Matplotlib Documentation* - <https://matplotlib.org/stable/contents.html>
- *Seaborn Documentation* - <https://seaborn.pydata.org/>
- *MkDocs Documentation* - <https://www.mkdocs.org/>
- Mini curso **Pandas**
- Mini curso **Data Visualization**

<div class="cards-grid shortcuts portfolio-list">

  <div class="card">
    <h3>Práctica 1 - Exploración y limpieza</h3>
    <p>Análisis exploratorio, chequeos básicos y validación de supuestos.</p>
    <p class="actions">
      <span class="pill">Completado</span>
      <a class="md-button md-button--primary" href="Práctica1/">Abrir</a>
    </p>
  </div>

  <div class="card">
    <h3>Práctica 3 - EDA Multi-fuentes y Joins - Fill in the Blanks</h3>
    <p>Análisis y visualizaciones a partir de caminos.</p>
    <p class="actions">
      <span class="pill">Completado</span>
      <a class="md-button md-button--primary" href="Práctica3/">Abrir</a>
    </p>
  </div>

  <div class="card">
    <h3>Práctica 4 - EDA Netflix con Visualizaciones - Fill in the Blanks</h3>
    <p>Visualizaciones y narrativa con foco en decisiones contextualizadas a la plataforma.</p>
    <p class="actions">
      <span class="pill">Completado</span>
      <a class="md-button md-button--primary" href="Práctica4/">Abrir</a>
    </p>
  </div>

</div>

---

## UT2: Calidad & Ética

**Al completar UT2, el estudiante será capaz de:**

- Distinguir entre **MCAR, MAR y MNAR** en datasets reales
- Detectar patrones de **missing data** y **outliers**
- Aplicar estrategias de **imputación** apropiadas según el contexto
- Implementar **pipelines de limpieza** reproducibles
- Prevenir **data leakage** usando **validación cruzada** apropiada
- Identificar y mitigar **sesgo** en datasets históricos
- Evaluar **fairness** usando métricas estándar
- Documentar **decisiones éticas** en el tratamiento de datos

**Lecturas mínimas (Evaluación el 03/09):**

- Zheng, A., & Casari, A. (2018). *Feature Engineering for Machine Learning*. O’Reilly Media.  
  1. Cap. 1 (**ML Pipeline**)  
  2. Cap. 2 (**Fancy Tricks with Simple Numbers**)  
  3. Cap. 4 (**Effects of Feature Scaling**)
- *Kaggle Data Cleaning* — <https://www.kaggle.com/learn/data-cleaning>
- *Kaggle Intermediate ML — Data Leakage* — <https://www.kaggle.com/code/alexisbcook/data-leakage>
- *Kaggle Intro to AI Ethics — Identifying Bias in AI; AI Fairness* —  
  <https://www.kaggle.com/learn/intro-to-ai-ethics>

**Lecturas Totales:**

- *Google ML Fairness* — <https://developers.google.com/machine-learning/crash-course/fairness>
- *Fairlearn Documentation* — <https://fairlearn.org/>
- *Pandas Missing Data Documentation* —  
  <https://pandas.pydata.org/docs/user_guide/missing_data.html>


<div class="cards-grid shortcuts portfolio-list">

  <div class="card">
    <h3>Práctica 5 - Missing Data Detective </h3>
    <p>Aprender a detectar y analizar datos faltantes (MCAR, MAR, MNAR).</p>
    <p class="actions">
      <span class="pill">Completado</span>
      <a class="md-button md-button--primary" href="Práctica5/">Abrir</a>
    </p>
  </div>

  <div class="card">
    <h3>Práctica 6 - Feature Scaling & Anti-Leakage </h3>
    <p> Generación de un Pipeline mediante Exploración Abierta.</p>
    <p class="actions">
      <span class="pill">Completado</span>
      <a class="md-button md-button--primary" href="Práctica6/">Abrir</a>
    </p>
  </div>

  <div class="card">
    <h3>Práctica 7 - Detectar y Corregir Sesgo con Fairlearn </h3>
    <p> Detección del sesgo en tres datasets distintos.</p>
    <p class="actions">
      <span class="pill">Completado</span>
      <a class="md-button md-button--primary" href="Práctica7/">Abrir</a>
    </p>
  </div>

</div>

---

## UT3: Feature Engineering

**Al completar UT3, el estudiante será capaz de:**

- Crear **features** derivadas relevantes según el dominio.
- Aplicar técnicas avanzadas de **encoding categórico**.  
- Manejar **variables de alta cardinalidad** efectivamente. 
- Implementar **PCA** para reducción dimensional.  
- Interpretar **componentes principales** y varianza explicada.  
- Construir **pipelines** de feature engineering escalables. 

**Lecturas mínimas (Evaluación el 01/10):**

- Zheng, A., & Casari, A. (2018). *Feature Engineering for Machine Learning*. O’Reilly Media.  
1. Cap. 2 (Fancy Tricks with Simple Numbers) - Transformaciones numéricas, binarización, interacciones  
2. Cap. 5 (Categorical Variables) - One-hot, label, ordinal y target encoding  
3. Cap. 6 (Dimensionality Reduction) - PCA, feature selection, curse of dimensionality 
- [Kaggle Feature Engineering - Curso Completo](https://www.kaggle.com/learn/feature-engineering)

**Lecturas Totales:**

**Scikit-learn Preprocessing & Encoders:**

- [Encoding Categorical Features](https://scikit-learn.org/stable/modules/preprocessing.html#encoding-categorical-features)  
- [PCA](https://scikit-learn.org/stable/modules/decomposition.html#pca)  
- [ColumnTransformer & Pipeline](https://scikit-learn.org/stable/modules/compose.html) 


<div class="cards-grid shortcuts portfolio-list">

  <div class="card">
    <h3>Práctica 8 - Feature Engineering con Pandas </h3>
    <p> Creación de Features derivadas que capturen patrones no obvios de los datos.</p>
    <p class="actions">
      <span class="pill">Completado</span>
      <a class="md-button md-button--primary" href="Práctica8/">Abrir</a>
    </p>
  </div>

  <div class="card">
    <h3>Práctica 9 - Encoding Avanzado y Target Encoding </h3>
    <p> Comparar diferentes técnicas de encoding para maximizar la precisión del modelo de clasificación.</p>
    <p class="actions">
      <span class="pill">Completado</span>
      <a class="md-button md-button--primary" href="Práctica9/">Abrir</a>
    </p>
  </div>
</div>




<div class="cards-grid shortcuts portfolio-list">

  <div class="card">
    <h3>Práctica 10 - PCA y Feature Selection </h3>
    <p> Comparar PCA vs Feature Selection, evaluando trade-offs en contexto de negocios.</p>
    <p class="actions">
      <span class="pill">Completado</span>
      <a class="md-button md-button--primary" href="Práctica10/">Abrir</a>
    </p>
  </div>

  <div class="card">
    <h3>Práctica 11 - Temporal Feature Engineering con Pandas</h3>
    <p> Implementación temporal feature engineering con datos transaccionales de e-commerce.</p>
    <p class="actions">
      <span class="pill">Completado</span>
      <a class="md-button md-button--primary" href="Práctica11/">Abrir</a>
    </p>
  </div>

</div>


---

## UT4: Datos Especiales

## 🎯 Objetivos de aprendizaje

Al completar **UT4: Datos Especiales**, el estudiante será capaz de:

- Manipular datos **geoespaciales** utilizando *GeoPandas* y *Shapely*.
- Extraer **features de audio** (*MFCC*) e **imagen** (*keypoints*) con librerías especializadas.
- Construir **pipelines para datos no estructurados** e integrarlos en sistemas de analítica avanzada.

## 📚 Lecturas mínimas  
**Evaluación el 05/11**

- Brust, A. V. (2023). *Ciencia de Datos para Gente Sociable* – Cap. 6: Información geográfica y mapas.  
- [Kaggle Geospatial Analysis](https://www.kaggle.com/learn/geospatial-analysis)  
- [GeoPandas Documentation – User Guide](https://geopandas.org/en/stable/docs/user_guide.html) *(Introduction, CRS, Plotting)*  
- [librosa Documentation – Tutorial + feature.mfcc](https://librosa.org/doc/main/feature.html#mfcc)
- [OpenCV Documentation – Feature Detection (SIFT y ORB)](https://docs.opencv.org/master/d7/d00/tutorial_meanshift.html)


<div class="cards-grid shortcuts portfolio-list">

  <div class="card">
    <h3>Próximamente</h3>
    <p>Las prácticas faltantes de esta unidad se publicarán aquí cuando estén disponibles.</p>
    <p class="actions">
      <span class="pill">En preparación</span>
      <a class="md-button" href="../">Volver al índice</a>
    </p>
  </div>

</div>

---

## Utilidades

<div class="cards-grid shortcuts portfolio-list">

  <div class="card">
    <h3>Nueva entrada</h3>
    <p>Usá la plantilla base para crear una práctica numerada.</p>
    <p class="actions">
      <a class="md-button" href="plantilla/">Crear desde plantilla</a>
    </p>
  </div>

</div>

---

_Última actualización: {{ page.meta.date }}_



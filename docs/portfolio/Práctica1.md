---
title: "01 — Análisis Exploratorio del dataset Iris"
date: 2025-09-07
number: 1
status: "Completado"
tags: [EDA, Visualización, Estadística descriptiva, Iris]
notebook: https://colab.research.google.com/drive/1chwkGY58rcG1R15Nguavc-XTnVwCsA0s?usp=sharing
drive_viz: https://drive.google.com/drive/folders/1qglTzvqdFPrNMxUhH_MtQFcRrafXEG7x?usp=sharing
dataset: "Iris (Fisher 1936) — scikit-learn / UCI"
time_est: "2 h 30 m"
time_spent: "—"
---

# {{ page.meta.title }}

<span class="pill">{{ page.meta.status }}</span>
<span class="pill">#{{ page.meta.number }}</span>
{% if page.meta.tags %}{% for t in page.meta.tags %}<span class="pill">{{ t }}</span>{% endfor %}{% endif %}

!!! abstract "Resumen ejecutivo"
    **Objetivo:** familiarizarme con el flujo del portafolio y caracterizar *Iris* para el modelado posterior.  
    **Datos:** 150 filas, 4 rasgos continuos (cm), 3 especies balanceadas (50 c/u).  
    **Hallazgos:** *setosa* separa casi perfecta con **petal_length < ~2.0**; **petal_length–petal_width** con correlación muy alta; las variables de **pétalo** concentran la señal.  
    **Resultado:** notebook reproducible en Colab, visualizaciones en Drive y lineamientos para baseline.


**Enlaces rápidos:**  
[Consigna Práctica 1](https://juanfkurucz.com/ucu-id/ut1/01-exploracion-iris/)

---

## Contexto
Actividad inicial del curso (EDA clásico sobre *Iris*). El objetivo es entender distribución, correlaciones y separabilidad por especie, además de **ordenar el flujo de trabajo** (repo + Drive) para reutilizarlo en próximas prácticas.

## Objetivos
- [x] Familiarizar el **flujo del portafolio** (estructura + publicación).  
- [x] Comprender las características principales del dataset.  
- [x] Identificar patrones/relaciones y posibles outliers.  
- [x] Organizar outputs en **carpetas de Google Drive** con nombres consistentes.

---

## Actividades (con tiempos estimados)

| Actividad                                   | Estimado | Real | Nota |
|---|---:|---:|---|
| Configurar repositorio                      | 30 m | **28 m** | Estructura mínima (`docs/`, `assets/`), `mkdocs.yml`, activación de GitHub Pages y verificación de workflow. Se agregó `extra.css` y variables de color. |
| Crear primera entrada                       | 30 m | **32 m** | Front-matter con metadatos, **pills** de estado/etiquetas, admoniciones y bloque “Resumen ejecutivo”. Ajustes de layout para que sea consistente con el sitio. |
| Google Colab                                | 10 m | **12 m** | Montaje de Drive, creación de carpetas `01/figs`, `01/tables`, helper para `savefig` con ruta absoluta de Drive. |
| Investigar el dataset                       | 10 m | **12 m** | Lectura de ficha UCI y `sklearn` (nº de filas, features, clases). Decisiones: renombrar a `snake_case` y trabajar en cm. |
| Preguntas de negocio                        | 5 m  | **7 m**  | ¿Qué variables separan mejor? ¿Hay outliers por especie? ¿Qué baseline probar primero? |
| Carga de datos (1–2 vías)                   | 15 m | **14 m** | **sklearn** `load_iris(as_frame=True)` y lectura alternativa CSV. Mapeo de `target` → `species` y renombrado de columnas. |
| Chequeos básicos + data dictionary          | 15 m | **18 m** | `df.info()`, nulos=0, duplicados=0, `describe()` global y por especie. Se redactó un mini diccionario de datos. |
| Plausibilidad y rangos                      | 10 m | **9 m**  | Min/Max por variable (cm) y revisión de boxplots; no se detectaron valores imposibles. |
| Análisis estadístico                        | 15 m | **22 m** | Medias/DE por especie, correlaciones, efecto de `petal_*` vs `sepal_*`. Heurística inicial para *setosa*. |
| Visualizaciones (mínimas + opc.)            | 30 m | **38 m** | Hist/box por especie, scatter `(petal_length, petal_width)`, **pairplot**, heatmap de correlación. Guardado a `01/figs/` con nombres consistentes. |
| Insights y comunicación                     | 20 m | **24 m** | Redacción del resumen, **Decisiones clave**, **Próximos pasos** y limpieza final del notebook/figuras. |

> **Totales** — Estimado: **3 h 10 m** · Real: **3 h 36 m** · Δ: **+26 m** (**+13.6%**).  
> **Desvío principal:** tiempo extra en visualizaciones (exportar a Drive + estilado) y en la primera puesta a punto del layout de la entrada.

**Lecciones para la próxima práctica**

- Reutilizar el **notebook skeleton** con helpers (`savefig`, `set_style`) para reducir fricción.  
- Mantener una **lista de figuras obligatorias** (3–4) y dejar el resto como opcional.  
- Completar la tabla de **tiempos reales** al terminar cada bloque (no al final).


---

## Desarrollo

- **Carga y schema:** 150 observaciones; 4 numéricas (`sepal_length`, `sepal_width`, `petal_length`, `petal_width`) y `species`. Sin nulos ni duplicados.  
- **Univariado:** variables de **pétalo** muestran mayor discriminación entre clases que las de sépalo.  
- **Bivariado:** `(petal_length, petal_width)` exhibe fronteras claras; *setosa* aislada con umbral simple.  
- **Correlaciones:** **muy alta** entre `petal_length` y `petal_width`.  
- **Estructura de trabajo:** outputs en Drive por práctica y tipo (figuras, tablas, reportes).

<details class="md-details">
  <summary><strong>Paso a paso (ejecución)</strong></summary>

  <ol>
    <li><strong>Preparar entorno</strong>
      <ul>
        <li>Montar Drive en Colab y definir rutas (<code>/content/drive/MyDrive/IA-Portfolio/01/</code>).</li>
      </ul>
    </li>
    <li><strong>Cargar datos</strong>
      <ul>
        <li><code>sklearn.datasets.load_iris(as_frame=True)</code> → <code>DataFrame</code>.</li>
        <li>Renombrar a <code>snake_case</code> y mapear <code>target</code> → <code>species</code>.</li>
      </ul>
    </li>
    <li><strong>Chequeos básicos</strong>
      <ul>
        <li><code>df.info()</code>, <code>df.isna().sum()</code>, duplicados y dimensiones.</li>
      </ul>
    </li>
    <li><strong>EDA univariado/bivariado</strong>
      <ul>
        <li>Histogramas/boxplots; scatter <code>(petal_length, petal_width)</code>; pairplot.</li>
      </ul>
    </li>
    <li><strong>Correlaciones y regla</strong>
      <ul>
        <li>Heatmap; regla <code>petal_length &lt; ~2.0</code> para <em>setosa</em>.</li>
      </ul>
    </li>
    <li><strong>Guardado de artefactos</strong>
      <ul>
        <li>Figuras <code>01/figs/</code>, tablas <code>01/tables/</code>, notas <code>01/notes/</code>.</li>
      </ul>
    </li>
  </ol>
</details>



---

## Métricas / Indicadores exploratorios

| Indicador                                  | Valor / Observación |
|---|---|
| Clases                                     | 3 (balanceadas: 50, 50, 50) |
| Nulos / Duplicados                         | 0 / 0 |
| Corr(`petal_length`, `petal_width`)        | Muy alta |
| Heurística para *setosa*                   | `petal_length < ~2.0` (separa casi perfecto) |

!!! tip "Criterios de aceptación"
    - [x] Dataset auditado y sin problemas de calidad.  
    - [x] Al menos una **regla heurística** útil documentada.  
    - [x] Visualizaciones exportadas y rutas reproducibles.

---

## Diccionario de datos (plausibilidad)
| Variable       | Unidad | Rango típico aprox. | Nota |
|---|:---:|:---:|---|
| `sepal_length` | cm | 4.3–7.9 | númerico continuo |
| `sepal_width`  | cm | 2.0–4.4 | númerico continuo |
| `petal_length` | cm | 1.0–6.9 | númerico continuo |
| `petal_width`  | cm | 0.1–2.5 | númerico continuo |
| `species`      | —  | {setosa, versicolor, virginica} | categórica |

---

## Evidencias

-  [**Práctica 1 (Colab):**](https://colab.research.google.com/drive/1chwkGY58rcG1R15Nguavc-XTnVwCsA0s?usp=sharing)
-  [**Visualizaciones (Drive):**](https://drive.google.com/drive/folders/1qglTzvqdFPrNMxUhH_MtQFcRrafXEG7x?usp=sharing) 

<div class="cards-grid media">

  <div class="card">
    <img src="../../assets/pairplot.png" alt="Pairplot Iris" loading="lazy">
    <div class="caption">
      Pairplot — todas las combinaciones
      <small><a href="{{ page.meta.drive_viz }}" target="_blank">Abrir carpeta</a></small>
    </div>
  </div>

  <div class="card">
    <img src="../../assets/hist_petal_length.png" alt="Histograma por largo de pétalos" loading="lazy">
    <div class="caption">
      Histograma <code>petal_length</code>
      <small>Separabilidad clara por especie</small>
    </div>
  </div>

  <div class="card">
    <img src="../../assets/corr.png" alt="Heatmap correlaciones" loading="lazy">
    <div class="caption">
      Heatmap de correlaciones
      <small>Variables de pétalo dominan la señal</small>
    </div>
  </div>

</div>


---

## Decisiones clave (ADR-lite)
- **Variables foco:** priorizar **pétalo** (mejor señal).  
- **Normalización:** diferida a modelado; evaluar impacto en modelos lineales.  
- **Semilla:** fijar para reproducibilidad (documentar en baseline).  
- **Estructura de artefactos:** Drive por práctica/tipo (consistente con futuras entradas).

!!! warning "Riesgos / Supuestos"
    - **Supuesto**: distribución estable y sin *leakage*.  
    - **Riesgo**: confundir correlación global con separabilidad por clase.  
    - **Mitigación**: métricas por clase y validación estratificada.

--- 

## Reproducibilidad
- Entorno: `python 3.11`; libs: `pandas`, `matplotlib`, `seaborn`, `scikit-learn`.  
- Cómo correr un EDA mínimo:

    ```bash
    pip install -q pandas matplotlib seaborn scikit-learn
    ```

    ```python
    from sklearn.datasets import load_iris
    import pandas as pd, seaborn as sns
    import matplotlib.pyplot as plt

    iris = load_iris(as_frame=True)
    df = iris.frame
    df['species'] = iris.target

    sns.scatterplot(data=df, x='petal length (cm)', y='petal width (cm)', hue='species')
    plt.tight_layout()
    plt.show()
    ```


---

!!! note "Reflexión"
    Lo más desafiante y valioso fue integrar **Google Drive** al flujo, dejando artefactos organizados y persistentes.  
    El repaso de **pandas / numpy / matplotlib / seaborn** reforzó las bases del EDA.  
    Confirmé que el valor está en **pétalos** y que con una regla simple ya se separa bien *setosa*.  
    El siguiente reto es cuantificar correctamente **versicolor vs. virginica** con métricas por clase.


## Próximos posibles pasos, cosas a mejorar en este u otros cursos:

- [x] Entrenar **Logistic Regression** y **KNN** con `train/test` estratificado.  
- [ ] Reporte de clasificación (macro/micro) + **curvas ROC/PR** por clase.  
- [ ] Comparar **normalización vs. sin normalización** (impacto en LR/KNN).  

## Referencias Particulares

- Enunciado: <https://juanfkurucz.com/ucu-id/ut1/01-exploracion-iris/>  
- Dataset: `sklearn.datasets.load_iris()` (Fisher, 1936)  
- UCI Machine Learning Repository — *Iris*


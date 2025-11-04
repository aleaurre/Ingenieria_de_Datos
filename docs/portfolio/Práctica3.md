---
title: "03 - EDA de Netflix con pandas"
date: 2025-09-07
number: 3
status: "Completado"
tags: [EDA, pandas, Visualización, Limpieza de datos, Netflix]
notebook: ../../evidencias/Aurrecochea-Practica4.ipynb
drive_viz: —
dataset: "Netflix Titles — Kaggle"
time_est: "3 h"
time_spent: "—"
---

# {{ page.meta.title }}

<span class="pill">{{ page.meta.status }}</span>
<span class="pill">#{{ page.meta.number }}</span>
{% if page.meta.tags %}{% for t in page.meta.tags %}<span class="pill">{{ t }}</span>{% endfor %}{% endif %}

!!! abstract "Resumen ejecutivo"
    **Objetivo:** realizar un **análisis exploratorio** del catálogo de Netflix: calidad de datos, nulos/duplicados, patrones por **tipo** (Movie/TV Show), **tiempo**, **país** y **géneros/ratings**.  
    **Datos:** dataset público “Netflix Titles” (Kaggle) con metadatos por título (director, cast, country, `date_added`, `release_year`, `rating`, `duration`, `listed_in`, etc.).  
    **Hallazgos (síntesis):** distribución asimétrica por **país** (concentración en pocos países), clara diferencia Movie/TV Show en **duración**, y “chorro” de altas **post-2015**; fuerte multi-etiquetado en `listed_in` que exige **explosión** previa a rankings.  
    **Resultado:** notebook reproducible en `docs/evidencias/`, figuras embebibles y chequeos de plausibilidad documentados.

**Enlaces rápidos:**  
[Consigna Práctica 3 — “EDA Netflix con Visualizaciones”](https://juanfkurucz.com/ucu-id/ut1/03-eda-netflix-pandas/)

---

## Contexto
Parte de la **UT1**: ejercitar un EDA completo con **pandas** y visualizaciones básicas (**matplotlib/seaborn**) para consolidar criterio de limpieza, **detección de outliers**, análisis temporal y de **géneros/ratings**, con foco en comunicar hallazgos accionables.

## Objetivos
- [x] Cargar y auditar el dataset (schema, nulos, duplicados, tipos y memoria).  
- [x] Analizar **tipos** (Movie/TV Show) con proporciones y conteos.  
- [x] Analizar **tendencias temporales** (por `release_year` y/o `date_added`).  
- [x] Analizar **geografía** (país de producción) y sesgos de cobertura.  
- [x] Analizar **géneros** (`listed_in`) y **ratings** (clasificación por edades).  
- [x] Documentar **outliers** y decisiones de limpieza.  

---

## Actividades (con tiempos estimados)

| Actividad                                             | Estimado | Real | Nota |
|---|---:|---:|---|
| Setup + carga de datos                                | 20 m | 25 m | Lectura CSV, estandarización de nombres a `snake_case`. |
| Auditoría de calidad (nulos/duplicados/tipos)         | 25 m | 20 m | Conteo y % de nulos, `drop_duplicates`, `astype` selectivo. |
| Tipos de contenido (conteos + pie/donut + countplot)  | 20 m | 20 m | Comparativa Movies vs TV Shows (porcentaje y absolutos). |
| Tendencia temporal                                    | 25 m | 30 m | Series por año; opcional: por mes vía `date_added`. |
| Geografía                                             | 25 m | 20 m | Limpieza de `country` (split/explode) y top países. |
| Géneros y ratings                                     | 30 m | 35 m | `listed_in` explode → ranking; distribución de `rating`. |
| Outliers y anomalías                                  | 20 m | 20 m | Años sospechosos, títulos duplicados, longitudes extremas. |
| Síntesis visual + narrativa                           | 15 m | 15 m | Selección de 3–5 gráficos y breve interpretación. |

> **Totales** — Estimado: **3 h** · Real: **3 h 10 m**.

---

## Desarrollo

- **Carga & schema**: lectura del CSV; normalización de columnas a `snake_case`; casting de fechas/numéricos cuando aplica.  
- **Quality checks**: tabla de **nulos** y **duplicados**; verificación de rangos plausibles (`release_year`, `duration`).  
- **Tipos**: conteos y proporciones Movie/TV Show con `value_counts()`/`normalize=True`.  
- **Temporal**: distribución por `release_year` (y opcional `date_added`), con énfasis en el período **2000+**.  
- **Geográfico**: `country` **explode** para evitar doble conteo; ranking top-N y cobertura “long tail”.  
- **Géneros/ratings**: `listed_in` **explode → strip → value_counts**; hist/stacked bar para ratings.  
- **Outliers**: detección en años (muy antiguos o futuros), títulos repetidos y **longitud de título** como proxy textual.


---

## Métricas / Indicadores exploratorios
| Indicador                                       | Valor / Observación |
|---|---|
| Nulos / Duplicados                              | Reportados en notebook |
| Proporción Movie vs TV Show                     | Reportada en notebook |
| Pico temporal (altas agregadas)                 | Reportado en notebook |
| Top países (catálogo por país)                  | Top-N tras **explode** |
| Géneros dominantes (`listed_in`)                | Ranking tras **explode** |
| Ratings (clasificación etaria)                  | Distribución por etiqueta |

!!! tip "Criterios de aceptación"
    - [x] Dataset auditado (nulos/duplicados/tipos) y visualizado.  
    - [x] Análisis por **tipo**, **tiempo**, **país**, **géneros** y **ratings**.  
    - [x] Outliers identificados con al menos un **boxplot** o **histograma**.  
    - [x] Evidencia reproducible en el **notebook** del repo.

---

## Diccionario de datos (plausibilidad)
| Variable        | Tipo        | Nota |
|---|:---:|---|
| `show_id`       | str/id      | Identificador único del título |
| `type`          | cat         | {Movie, TV Show} |
| `title`         | str         | Nombre del contenido |
| `director`      | str         | Director/a (puede ser nulo) |
| `cast`          | str         | Lista de protagonistas (string) |
| `country`       | str/list    | País(es) de producción (separados por coma) |
| `date_added`    | date/str    | Fecha de alta al catálogo |
| `release_year`  | int         | Año de lanzamiento |
| `rating`        | cat         | Clasificación por edades |
| `duration`      | str / int   | Minutos (Movie) o temporadas (TV Show) |
| `listed_in`     | str/list    | Géneros/categorías (separados por coma) |
| `description`   | str         | Sinopsis breve |

---

## Evidencias

-  [**Notebook (.ipynb)**](../../evidencias/Aurrecochea-Práctica3.ipynb)

---

## Decisiones clave (ADR-lite)
- **Explode primero, agrupar después** en `country` y `listed_in` para evitar **doble conteo**.  
- **Temporal**: usar `release_year` como base y **triangular** con `date_added` si se requiere “fecha de incorporación”.  
- **Ratings**: homogenizar etiquetas (trim/upper) antes de contar; agrupar variantes equivalentes.  
- **Outliers**: criterio 1-SE sobre `release_year` + inspección manual de títulos extremos.

---

## Reproducibilidad
- Entorno: `python 3.11`; libs: `pandas`, `matplotlib`, `seaborn`.  
- Ejecutable 100% en el **.ipynb** del repo (`docs/evidencias/…`).  
- Gráficos generados **sin estilos externos**; paletas por defecto para portabilidad.

---

!!! note "Reflexión"
    Esta práctica consolidó **disciplina EDA**: primero **auditar**, luego **visualizar**, y recién ahí **concluir**.  
    La mayor trampa fue el **multi-valor** en `country`/`listed_in`: sin **explode**, los rankings engañan.  
    También aprendí que `date_added` cuenta una historia distinta a `release_year` (negocio ≠ producción), útil para pensar **políticas de adquisición** y **estrategias por país/género**.

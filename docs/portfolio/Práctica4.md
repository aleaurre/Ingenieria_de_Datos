---
title: "04 — EDA Multifuentes + Joins (NYC Taxi, Zones, Calendar)"
date: 2025-09-07
number: 4
status: "Completado"
tags: [EDA, Pandas, Joins, GroupBy, Parquet, CSV, JSON, Performance]
notebook: ../../evidencias/Aurrecochea-Practica4.ipynb
drive_viz: —
dataset: "NYC Yellow Taxi (enero 2023) + NYC Taxi Zones + Calendar"
time_est: "3 h 05 m"
time_spent: "3 h 02 m"
---

# {{ page.meta.title }}

<span class="pill">{{ page.meta.status }}</span>
<span class="pill">#{{ page.meta.number }}</span>
{% if page.meta.tags %}{% for t in page.meta.tags %}<span class="pill">{{ t }}</span>{% endfor %}{% endif %}

!!! abstract "Resumen ejecutivo"
    **Objetivo:** integrar **múltiples fuentes** (Parquet, CSV, JSON), enriquecer viajes con **geografía** y **eventos**, y producir **agregados por borough** y por **día especial**.  
    **Datos:** Yellow Taxi (2023-01, **3,066,766** viajes), NYC Taxi Zones (**265** zonas) y calendar de eventos.  
    **Hallazgos:** **match_rate 99.6%** en el join con zonas; **zone_coverage 97.7%** (259/265 zonas con uso). Picos de demanda a **18:00**, **17:00** y **15:00**. No hubo **días especiales** en el período (calendar sin eventos para 2023-01).  
    **Resultado:** notebook reproducible con joins encadenados, KPIs y técnicas de **performance** (tipificación y muestreo).

**Enlaces rápidos:**  
[Consigna — “Tarea 4: EDA Multifuentes y Joins”](https://juanfkurucz.com/ucu-id/ut1/04-eda-multifuentes-joins/)

---

## Contexto
Análisis integrado a escala: viajes (~3M/mes) + **lookup** de zonas (borough/zone) + calendario de eventos para contrastar volumen y montos. Enfatiza **calidad de joins**, **normalización** y **eficiencia** en pandas.

## Objetivos
- [x] Cargar datos desde **Parquet/CSV/JSON**.  
- [x] **Normalizar** nombres y tipos; derivar `pickup_date`.  
- [x] Join **trips + zones** (*left*) y verificación de nulos/cobertura.  
- [x] Join **+ calendar** y bandera `is_special_day`.  
- [x] **Agregados** por **borough** y **borough × is_special_day**.  
- [x] Reporte de **match_rate**, **zone_coverage** y picos horarios.

---

## Actividades (con tiempos estimados)

| Actividad                                                  | Estimado | Real | Nota |
|---|---:|---:|---|
| Setup + lectura multifuentes (Parquet/CSV/JSON)            | 25 m | **24 m** | Validación de esquemas y rutas. |
| Normalización y tipos (snake_case, fechas, dtypes)         | 20 m | **19 m** | Derivación `pickup_date`; `astype` de IDs. |
| Join 1: `trips ⟶ trips+zones` (left)                       | 20 m | **19 m** | Chequeo de nulos y columnas clave. |
| Join 2: `+ calendar` (left) + `is_special_day`             | 20 m | **19 m** | Alineación por fecha; sin eventos en 2023-01. |
| Agregados por borough (conteos/medias/medianas/std)        | 20 m | **20 m** | `groupby('borough')` con métricas. |
| Agregados borough × día especial                           | 20 m | **19 m** | Comparativa normal vs especial (solo normal). |
| Técnicas para datasets grandes (muestreo/tipificación)     | 25 m | **24 m** | *Downcast* y muestra estratificada para plots. |
| Validación/QA (nulos clave, integridad)                    | 15 m | **14 m** | Revisiones de cobertura/match. |
| Síntesis visual + narrativa                                | 20 m | **24 m** | Selección de 5 figuras y captions. |

> **Totales** — Estimado: **3 h 05 m** · Real: **3 h 02 m** · Δ: **−3 m** (**−1.6%**).

---

## Desarrollo

- **Fuentes:** `yellow_tripdata_2023-01.parquet` (viajes), `taxi+_zone_lookup.csv` (zones/borough), `calendar.json` (eventos).  
- **Joins:**  
  1) `trips ⨝ zones` (*left*, `pulocationid`→`locationid`),  
  2) `trips_with_zones ⨝ calendar` (*left*, `pickup_date`→`date`) + `is_special_day`.  
- **Performance:** muestreo estratificado para visualizaciones (10,000 filas ≈ 0.3%) y *downcast* de `dtypes`.

<details class="md-details">
  <summary><strong>Paso a paso (ejecución)</strong></summary>
  <ol>
    <li><strong>Lectura</strong> Parquet/CSV/JSON; vistas rápidas y memoria.</li>
    <li><strong>Normalización</strong> de columnas y fechas; tipos para join.</li>
    <li><strong>Join trips+zones</strong> y validación de nulos/cobertura.</li>
    <li><strong>Join + calendar</strong> y creación de <code>is_special_day</code>.</li>
    <li><strong>Agregados</strong> por borough y borough×día especial.</li>
    <li><strong>Performance & QA</strong>: muestreo, métricas de match, chequeos finales.</li>
  </ol>
</details>

---

## Métricas / Indicadores

| Indicador                               | Valor |
|---|---|
| **Viajes totales**                      | **3,066,766** |
| **Distancia promedio**                  | **3.85** millas |
| **Tarifa promedio**                     | **USD 27.02** |
| **matched_zones** (trips↔zones)         | **3,055,808** |
| **match_rate**                          | **99.6%** |
| **unique_zones_used / total_zones**     | **259 / 265** |
| **zone_coverage**                       | **97.7%** |
| **Picos horarios (viajes)**             | **18:00 (215,889)** · **17:00 (209,493)** · **15:00 (196,424)** |
| **Top 3 boroughs (viajes)**             | **Manhattan (2,715,369)** · **Queens (286,645)** · **Unknown (40,116)** |
| **Días especiales**                     | No hubo eventos → **solo “Día Normal”** |

> **Borough (medias a modo de ejemplo)** — Manhattan: *dist* 3.19 · *importe* 23.86 | Queens: 8.81 · 49.50 | Staten Island: 17.96 · 90.31 | Unknown: 3.02 · 27.90

!!! tip "Criterios de aceptación"
    - [x] Lectura **multifuentes** y normalización coherente.  
    - [x] **Joins** correctos con verificación de nulos y cobertura.  
    - [x] **Agregados** por borough y comparación por **días especiales**.  
    - [x] Reporte de **performance** (muestra/ahorro de memoria).

---

## Evidencias
- [**Notebook (.ipynb) dentro del sitio**](../../evidencias/Aurrecochea-Practica4.ipynb)

---

## Decisiones clave (ADR-lite)
- **Join**: `left` en ambos pasos para conservar todos los viajes y medir *match rate*.  
- **Temporal**: `pickup_date` (fecha sin hora) para alinear con `calendar.date`.  
- **Optimización**: *downcast* en IDs/contadores; muestreo para visualizaciones masivas.  
- **Validación**: chequeos de nulos y conteos por borough tras cada join.

---

## Reproducibilidad
`python 3.11`; `pandas`, `numpy`, `matplotlib`, `seaborn`.  
El notebook incluye logs de ejecución y resumen final con KPIs.

---

!!! note "Reflexión"
    Integrar **múltiples fuentes** cambia el juego: la señal aparece recién cuando juntamos **viajes + zonas + eventos**. Lo más desafiante fue mantener **calidad de join** a gran escala (**IDs válidos, fechas consistentes, cobertura por zona**) sin romper performance; la estrategia de **tipificación + muestreo** funcionó bien y deja la base lista para escalar a más meses u orígenes.
---
title: "06 — Feature Scaling & Anti-Leakage (Ames Housing)"
date: 2025-09-20
number: 6
status: "Completada"
tags: [Feature Scaling, StandardScaler, RobustScaler, MinMaxScaler, Anti-leakage, Pipeline, Cross-Validation, Yeo-Johnson, QuantileTransformer, Ames Housing]
notebook: docs/evidencias/Aurrecochea-Práctica6.ipynb
drive_viz: —
dataset: "Ames Housing — Kaggle"
time_est: "3 h 30 m"
time_spent: "—"
---

# {{ page.meta.title }}

<span class="pill">{{ page.meta.status }}</span>
<span class="pill">#{{ page.meta.number }}</span>
{% if page.meta.tags %}{% for t in page.meta.tags %}<span class="pill">{{ t }}</span>{% endfor %}{% endif %}

!!! abstract "Resumen ejecutivo"
    **Objetivo:** evaluar la necesidad de escalado en Ames Housing; aplicar y comparar distintos escaladores (**MinMax**, **Standard**, **Robust**); probar transformaciones avanzadas (**log1p**, **Yeo-Johnson**, **QuantileTransformer**); y validar la importancia de usar *pipelines* para evitar **data leakage**.  
    **Scope:** selección de 5 *features* numéricas (`Lot Area`, `Overall Qual`, `Year Built`, `1st Flr SF`, `Gr Liv Area`) con *target* `SalePrice`.  
    **Resultado:** *Pipeline* recomendado = `log1p` en variables muy sesgadas → imputación (mediana / One-Hot) → `StandardScaler` → `LinearRegression`, con **R² = 0.776 ± 0.033 (5-fold)**.  

---

## Contexto
Los modelos sensibles a distancias o magnitudes (KNN, SVM, regresiones lineales con regularización) requieren escalado para garantizar estabilidad y comparabilidad. Esta práctica explora el impacto de diferentes transformaciones y la implementación de *pipelines* anti-leakage con validación cruzada.

## Objetivos
- [x] Detectar variables con escalas muy dispares.  
- [x] Evaluar transformaciones para reducir sesgo y *outliers*.  
- [x] Implementar y comparar distintos escaladores.  
- [x] Demostrar con métricas la importancia de anti-leakage.  
- [x] Validar resultados con **cross-validation** estable.  

---

## Actividades (con tiempos estimados)

| Actividad | Estimado | Real | Nota |
|---|---:|---:|---|
| Análisis inicial de escalas | 15 m | **20 m** | Ratio máx/min en cada feature. |
| Outliers y sesgo | 25 m | **35 m** | Conteo con IQR y Z-score; skewness. |
| Transformaciones avanzadas | 30 m | **40 m** | log1p, Yeo-Johnson, QuantileTransformer. |
| Comparación de escaladores | 30 m | **38 m** | Standard vs. MinMax vs. Robust. |
| Anti-leakage (demos) | 30 m | **42 m** | M1/M2 vs Pipeline + CV. |
| Validación final | 30 m | **37 m** | Cross-validation 5 folds; R², RMSE. |
| Informe y reflexión | 20 m | **25 m** | Redacción final y recomendaciones. |

> **Totales** — Estimado: **3 h 00 m** · Real: **3 h 57 m** · Δ: **+57 m** (**+31.7%**).

---

## Desarrollo

### 1) Análisis de escalas
- `Lot Area`: ratio máx/min ≈ **165.6** → escala más dispareja.  
- `SalePrice`: ratio ≈ **59**.  
→ Justifica escalado, sobre todo para modelos sensibles.

### 2) Outliers y sesgo
- En `SalePrice`: IQR detecta **137** outliers, Z-score **45**.  
- **Corrección aplicada:** los cambios en conteo tras el escalado no se deben al escalador, sino a trabajar sobre distinto *split*. Es clave comparar *outliers* siempre sobre el mismo subconjunto.

### 3) Transformaciones avanzadas
- `Lot Area`: skew ≈ **12.8** → con `log1p` baja a **–0.49**.  
- `Yeo-Johnson`: reduce skew de `SalePrice` a ≈ 0, y baja outliers (IQR 137→59, Z-score 45→20).  
- `QuantileTransformer`: normaliza bien distribuciones sesgadas.  
- **Normalizer/MaxAbs**: no útiles en este caso.

### 4) Anti-leakage (demos)
- **M1 (con leakage, single split):** RMSE ≈ 32185.  
- **M2 (sin leakage manual, single split):** RMSE ≈ 32287.  
- **M3 (pipeline + 5-fold CV):** RMSE ≈ 30651.  
→ Conclusión: el pipeline con CV no es “más optimista”, sino más **estable y honesto**.

### 5) Validación final
- Modelo: `StandardScaler` + `LinearRegression`.  
- **R² = 0.776 ± 0.033** (5 folds).  
- Baseline (DummyRegressor): mucho menor, confirma el valor agregado.

---

## Métricas / Indicadores

| Indicador | Valor/Observación |
|---|---|
| Ratio máx/min (`Lot Area`) | 165.6 |
| Skew inicial (`Lot Area`) | 12.8 |
| Skew tras log1p | –0.49 |
| Outliers IQR/Z-score (`SalePrice`) | 137 / 45 |
| Outliers tras Yeo-Johnson | 59 / 20 |
| RMSE M1/M2/M3 | 32185 / 32287 / 30651 |
| R² final (5-fold) | 0.776 ± 0.033 |

---

## Decisiones clave (ADR-lite)
- Escalado necesario en `Lot Area` y `SalePrice`.  
- `log1p` o Yeo-Johnson en variables con skew alto.  
- Mantener `StandardScaler` como opción por defecto.  
- Uso de *pipelines* anti-leakage con CV obligatorio.  
- Baseline (`DummyRegressor`) como punto de comparación.

!!! warning "Correcciones aplicadas"
    - La reducción de *outliers* al escalar no es real: se debe a cambios de muestra. Corregido en el análisis.  
    - La conclusión sobre “optimismo” fue ajustada: no es que el pipeline sea más optimista, sino que provee estimaciones más robustas.  
    - Se recomienda **imputación** y **One-Hot Encoding** en pipeline, en lugar de *drop* de filas con NaNs.

---

## Evidencias
- [**Notebook (.ipynb)**](../../evidencias/Aurrecochea-Práctica6.ipynb)
- [**Notebook (.ipynb) de Profundización en el análisis (BONUS)**](../../evidencias/Aurrecochea-Práctica6Bonus.ipynb)

<div class="cards-grid media">

  <div class="card">
    <img src="../../assets/Práctica6/boxplots.png" alt="Boxplots de variables numéricas" loading="lazy">
    <div class="caption">
      Boxplots de escalas
      <small>Comparación de magnitudes en Lot Area, SalePrice y otras features</small>
    </div>
  </div>

  <div class="card">
    <img src="../../assets/Práctica6/comparacionlogs.png" alt="Distribuciones antes y después de log1p" loading="lazy">
    <div class="caption">
      Transformación log1p
      <small>Reducción de skewness en Lot Area tras aplicar logaritmo</small>
    </div>
  </div>

  <div class="card">
    <img src="../../assets/Práctica6/histogramas.png" alt="Histogramas de detección de outliers" loading="lazy">
    <div class="caption">
      Histogramas de outliers
      <small>Comparación de detección por IQR y Z-score</small>
    </div>
  </div>

</div>

## Transformations & Scalers (evidencias)

<div class="cards-grid media">

  <div class="card">
    <img src="../../assets/Práctica6/FunctionTransformer.png" alt="FunctionTransformer con log1p aplicado a variables sesgadas" loading="lazy">
    <div class="caption">
      FunctionTransformer (log1p)
      <small>Para colas largas y valores 0; reduce skew sin desescalar magnitudes.</small>
    </div>
  </div>

  <div class="card">
    <img src="../../assets/Práctica6/YeoJohnson.png" alt="Power transform Yeo-Johnson para normalizar distribuciones" loading="lazy">
    <div class="caption">
      PowerTransformer — Yeo-Johnson
      <small>Normaliza incluso con ceros/negativos; útil cuando log1p no alcanza.</small>
    </div>
  </div>

  <div class="card">
    <img src="../../assets/Práctica6/QuantileTransformer.png" alt="QuantileTransformer hacia distribución Normal" loading="lazy">
    <div class="caption">
      QuantileTransformer (→ Normal)
      <small>Hace monotónica y ~Normal la variable; robusto a outliers, cambia distancias.</small>
    </div>
  </div>

  <div class="card">
    <img src="../../assets/Práctica6/MaxAbsScaler.png" alt="MaxAbsScaler escala por valor absoluto máximo" loading="lazy">
    <div class="caption">
      MaxAbsScaler
      <small>Escala a [-1,1] sin centrar; bueno para datos dispersos/sparse.</small>
    </div>
  </div>

  <div class="card">
    <img src="../../assets/Práctica6/NormalizerL2.png" alt="Normalizer L2 reescala filas a norma unitaria" loading="lazy">
    <div class="caption">
      Normalizer (L2)
      <small>Normaliza filas (vectores de muestra); útil en similitud/coseno, no para tabulares clásicos.</small>
    </div>
  </div>

</div>

---

## Comparación de transformers investigados

| Transformer | Tipo de transformación | Ventajas | Limitaciones / Cuándo NO usarlo |
|-------------|------------------------|----------|--------------------------------|
| **FunctionTransformer (log1p)** | Logarítmica simple (x → log(1+x)) | Fácil de interpretar, comprime colas largas, mantiene orden | Solo para x ≥ 0; no corrige totalmente asimetrías extremas |
| **Yeo-Johnson (PowerTransformer)** | Transformación de potencia paramétrica | Funciona con ceros y negativos, reduce skew, aproxima a Normal | Menos interpretable, puede distorsionar relaciones lineales |
| **QuantileTransformer (→ Normal)** | Monótona basada en cuantiles | Elimina asimetría, distribuye ~Normal, muy robusto a outliers | Distancias absolutas dejan de ser significativas; pierde interpretación original |
| **MaxAbsScaler** | Escala cada feature dividiendo por su valor absoluto máximo | Conserva dispersión relativa, útil en datos dispersos (*sparse*) | No centra en media 0, no corrige outliers, no afecta skew |
| **Normalizer (L2)** | Reescala cada fila a norma unitaria | Útil para similitud coseno / ML basado en ángulos | No es apropiado para tabulares clásicos; distorsiona magnitudes absolutas |

!!! tip "Síntesis"
    - Para variables **sesgadas** → `log1p` o `Yeo-Johnson`.  
    - Para variables con **colas extremas y outliers fuertes** → `QuantileTransformer`.  
    - Para **datos dispersos** (ej. texto vectorizado) → `MaxAbsScaler`.  
    - Para **métricas angulares** (ej. KNN con coseno) → `Normalizer L2`.  
    - Como **default general en tabulares** → `StandardScaler` (fuera de esta tabla, pero quedó como baseline final).



---

## Cuestionario y respuestas

1. **¿Qué variables requieren escalado y por qué?**  
   `Lot Area` (ratio 165.6) y `SalePrice` (ratio 59) tienen escalas muy dispares → necesario para modelos sensibles a magnitud.

2. **¿Qué pasa con los outliers al aplicar escalado?**  
   El escalado no elimina outliers; los cambios en conteo se debieron a trabajar sobre distinto *split*. Para reducir sesgo se recomienda usar log/Yeo-Johnson o winsorización.

3. **¿Qué transformaciones mejoraron más la forma de las distribuciones?**  
   - `log1p` en `Lot Area` (skew 12.8 → –0.49).  
   - `Yeo-Johnson` y `QuantileTransformer` en `SalePrice`, que bajaron outliers y normalizaron la distribución.

4. **¿Qué demostró el ejercicio de leakage?**  
   Que aplicar transformaciones antes del split genera estimaciones poco realistas. Solo con *pipelines* y CV se logra evaluar sin fuga de información.

5. **¿Qué protocolo de validación fue más confiable?**  
   El **Pipeline + 5-fold CV**, con R² = 0.776 ± 0.033. Más estable y menos dependiente de un único split.

6. **Recomendación final de pipeline**  
   - `log1p` en variables sesgadas.  
   - Imputación (mediana para numéricas, constante/One-Hot para categóricas).  
   - `StandardScaler`.  
   - Modelo lineal simple como baseline, extensible a KNN/SVR.  

---

!!! note "Reflexión"
    Esta práctica me permitió comprobar que **el escalado y las transformaciones deben justificarse con métricas** (ratio, skew, outliers). También confirmé que **el anti-leakage es innegociable**: un pipeline bien armado no solo garantiza reproducibilidad, sino que cambia radicalmente la validez de los resultados. 

---

## Próximos pasos
- [ ] Probar escaladores en KNN, SVR y Random Forest.  
- [ ] Evaluar `TransformedTargetRegressor` para aplicar log al target.  
- [ ] Medir con múltiples métricas (RMSE, MAE, MAPE).  
- [ ] Agregar winsorización para casos extremos en Lot Area.

---
## De cara a futuros proyectos:
- Voy a incorporar **TransformedTargetRegressor** para transformar también el *target* en datasets sesgados.  
- Planeo probar **combinaciones de transformers** (ej. log1p + RobustScaler) para casos con outliers extremos.  
- Voy a documentar cada decisión en un **ADR-lite** (mini registro de decisiones), lo que facilita replicar o defender el pipeline en entornos profesionales.  
- Finalmente, pienso incluir estas comparaciones en un **repositorio de “recetas de preprocesamiento”** propio, para reutilizarlas rápidamente en proyectos de machine learning.

## 🏆 MI CHECKLIST PERSONAL PARA PROYECTOS DE DATOS:

- [ ] 1. ¿Las features están en escalas muy diferentes?
- [ ] 2. ¿Mi proceso necesita escalado?  
- [ ] 3. ¿Hay outliers evidentes? → ¿RobustScaler?
- [ ] 4. ¿Datos muy sesgados? → ¿Log transform?
- [ ] 5. ¿Estoy usando Pipeline? → SIEMPRE (anti-leakage)
- [ ] 6. ¿Split ANTES de transformar? → OBLIGATORIO
- [ ] 7. ¿Cross-validation honesta? → Pipeline + CV
- [ ] 8. ¿Resultados realistas vs optimistas? → Detectar leakage
- [ ] 9. ¿Documenté mi elección de transformadores?
- [ ] 10. ¿Mi pipeline es reproducible?

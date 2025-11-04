---
title: "11 - Temporal Feature Engineering"
date: 2025-11-03
number: 11
status: "Completada"
tags: [Feature Engineering, Time Series, E-commerce, Lag Features, Rolling, Expanding, RFM, Calendar Encoding, Pandas, Data Leakage]
notebook: docs/evidencias/Aurrecochea-Práctica11.ipynb
drive_viz: —
dataset: "Online Retail (Kaggle / Reino Unido, 2010–2011)"
time_est: "2 h 30 m"
time_spent: "2 h 25 m"
---

# {{ page.meta.title }}
<span class="pill">{{ page.meta.status }}</span>
<span class="pill">#{{ page.meta.number }}</span>
{% if page.meta.tags %}{% for t in page.meta.tags %}<span class="pill">{{ t }}</span>{% endfor %}{% endif %}

!!! abstract "Resumen ejecutivo"
    **Objetivo:** aplicar técnicas de **ingeniería temporal de features** (*Temporal Feature Engineering*) sobre un dataset real de transacciones de e-commerce, utilizando `pandas` para generar *lags*, *rolling windows*, *expanding windows*, *RFM aggregations* y *calendar features*.  
    **Scope:** analizar la recurrencia de clientes, tendencias de compra y estacionalidades a partir de secuencias temporales individuales.  
    **Resultado:** se obtuvo un conjunto enriquecido de variables que describen la dinámica temporal del cliente, evitando *data leakage* mediante agrupamientos por usuario y desplazamientos controlados. Se identificaron patrones de frecuencia mensual, diversidad de productos y efectos de calendario sobre la demanda.

---

## Contexto general

Esta práctica corresponde a la **Unidad 3 (UT3-11)** del curso *Inteligencia de Datos*,  **“Temporal Feature Engineering”**, basada en la pauta:  [juanfkurucz.com/ucu-id/ut3/11-temporal-features-assignment](https://juanfkurucz.com/ucu-id/ut3/11-temporal-features-assignment/).
El dataset **Online Retail (Kaggle)** contiene **541 909 transacciones** entre diciembre de 2010 y diciembre de 2011. Cada registro representa una venta con identificador de cliente, producto, fecha y país, permitiendo analizar **secuencias de compra a nivel individual**. El objetivo principal es **extraer características temporales** que capturen hábitos de consumo, frecuencia de recompra, tendencias recientes y estacionalidad.

---

## Objetivos específicos

1. Implementar *Temporal Feature Engineering* con `pandas` sobre datos transaccionales.  
2. Generar variables de tipo **lag**, **rolling**, **expanding** y **ventanas temporales (7d–30d–90d)**.  
3. Calcular agregaciones **RFM** (Recency, Frequency, Monetary).  
4. Incorporar **features cíclicas y externas** (mes, día, feriados, indicadores económicos).  
5. Garantizar **validación temporal robusta** sin fuga de información (*no data leakage*).  

---

## Pauta del assignment

| Etapa | Descripción |
|:--|:--|
| **1. Carga y exploración del dataset** | Descarga automática desde Kaggle y verificación de estructura. |
| **2. Limpieza y preprocesamiento** | Eliminación de nulos, cancelaciones y valores negativos. |
| **3. Creación de features derivadas por orden** | Nivel de factura: `cart_size`, `order_total`, `days_since_prior_order`. |
| **4. Lags y ventanas móviles** | `shift()`, `rolling()`, `expanding()` agrupadas por `user_id`. |
| **5. RFM Aggregations** | Recency, Frequency, Monetary acumuladas. |
| **6. Time Windows (7d–30d–90d)** | Actividad reciente e histórica. |
| **7. Product Diversity** | Ratio de diversidad de productos por cliente. |
| **8. Calendar y External Features** | Encoding cíclico (sin/cos) y simulación de indicadores económicos. |

!!! quote "Criterios de aceptación"
    - Variables temporales correctamente ordenadas y sin *data leakage*.  
    - Uso de `.groupby()` + `.shift()` para generar lags independientes por usuario.  
    - Aplicación de *rolling* y *expanding windows* con visualización e interpretación.  
    - Construcción de métricas RFM y diversidad de productos.  
    - Incorporación de *calendar encoding* y análisis de estacionalidad.

---

## Preparación del dataset

Tras la autenticación de Kaggle y carga del CSV:

| Métrica | Valor |
|:--|--:|
| Filas iniciales | 541 909 |
| Filas luego de limpieza | 397 884 |
| Clientes únicos | 4 338 |
| Productos únicos | 3 665 |
| Rango temporal | 2010-12-01 → 2011-12-09 |
| Ventas totales | USD 8 911 407,90 |

> Se removieron facturas canceladas, precios negativos y clientes sin ID.  
> El dataset se ordenó por `user_id` y `order_date`, garantizando coherencia temporal.

!!! note "Estructura del dataset limpio"
    - Cada fila representa una **línea de transacción** (producto dentro de una orden).  
    - Las compras son **eventos irregulares**: los intervalos entre órdenes varían entre usuarios.  
    - Promedio: **4,27 órdenes por cliente** y **21,4 ítems por orden**.

---

## Lags – Comportamiento reciente

El uso de `.groupby('user_id').shift(n)` permitió generar **lags temporales** sin contaminación entre usuarios:

| Lag              | Interpretación                          |
| :--------------- | :-------------------------------------- |
| `lag_1`          | Intervalo entre las dos últimas compras |
| `lag_2`, `lag_3` | Frecuencia de compra a mediano plazo    |

!!! success "Resultados:"
    Los lags revelan consistencia temporal: clientes con compras frecuentes muestran valores pequeños y estables de `days_since_prior_order`. Este tipo de variable es clave para detectar **clientes regulares vs. esporádicos**.

---

## Rolling Windows – Tendencias recientes

Se calcularon promedios y desviaciones móviles sobre las tres últimas órdenes:

```python
orders_df['rolling_cart_mean_3'] = (
    orders_df.groupby('user_id')['cart_size']
    .shift(1)
    .rolling(window=3, min_periods=1)
    .mean())
```

![](../../assets/Práctica11/rollingmean_vs_cartsize.png)

<div class="cards-grid media">

  <div class="card">
    <alt="Rolling Mean vs Actual Cart Size" loading="lazy">
    <div class="caption">
      Rolling mean y desviación de tamaño de carrito.  
      <small>Captura fluctuaciones recientes en el comportamiento de compra.</small>
    </div>
  </div>

</div>

!!! success "Resultados:"
    Las *rolling windows* suavizan picos y permiten medir la **tendencia reciente**.
    Clientes con alta varianza presentan comportamiento impulsivo; los de baja varianza, hábitos consistentes.

---

## Expanding Windows – Comportamiento histórico

```python
orders_df['expanding_days_mean'] = (
    orders_df.groupby('user_id')['days_since_prior_order']
    .shift(1)
    .expanding(min_periods=1)
    .mean())
```

Las *expanding windows* acumulan estadísticas desde el inicio hasta la fecha actual:

| Tipo            | Horizonte   | Ejemplo de aplicación             |
| :-------------- | :---------- | :-------------------------------- |
| **Rolling (3)** | Corto plazo | Tendencia reciente                |
| **Expanding**   | Largo plazo | Fidelidad o retención del cliente |

!!! success "Resultados:"
    Las *expanding features* reflejan la estabilidad a largo plazo.
    En usuarios leales, el promedio histórico converge a un valor constante;
    en clientes nuevos o volátiles, varía ampliamente.

---

## RFM Aggregations – Recency, Frequency, Monetary

Se generaron las métricas clásicas:

| Variable                 | Definición                  | Insight                                                  |
| :----------------------- | :-------------------------- | :------------------------------------------------------- |
| `recency_days`           | Días desde la última compra | Clientes recientes muestran mayor propensión a recompra. |
| `frequency_total_orders` | Órdenes totales por cliente | Mide lealtad e historial de interacción.                 |
| `monetary_avg`           | Gasto promedio histórico    | Permite segmentar según ticket medio.                    |

![](../../assets/Práctica11/distributions.png)

**Correlaciones principales:**

| Relación             | Valor | Interpretación                                     |
| :------------------- | :---: | :------------------------------------------------- |
| Recency ↔ Monetary   | +0.26 | Clientes recientes tienden a gastar más.           |
| Frequency ↔ Monetary | −0.33 | Frecuencia alta → compras pequeñas pero regulares. |

> Este patrón es típico de comercios minoristas: pocos clientes concentran gran parte del gasto, mientras la mayoría realiza compras frecuentes de bajo monto.

---

## Time Windows (7d, 30d, 90d)

Se calcularon ventanas temporales móviles por usuario:

| Ventana | Órdenes promedio | Gasto promedio (USD) |
| :------ | ---------------: | -------------------: |
| 7 días  |             0.41 |                  295 |
| 30 días |             1.42 |                  923 |
| 90 días |             3.69 |                2 393 |

![](../../assets/Práctica11/ventas_temporal.png)

<div class="cards-grid media">
  <div class="card">
    <alt="Comparativa de ventanas temporales" loading="lazy">
    <div class="caption">
      Ventanas temporales (7d–30d–90d).  
      <small>La actividad reciente (7d) es baja, lo que evidencia comportamiento esporádico.</small>
    </div>
  </div>

</div>

!!! success "Resultados:"
    Las ventanas móviles revelan períodos de **reactivación** (usuarios que vuelven a comprar)
    y **desaceleración** (usuarios dormidos).
    Comparar `orders_7d` vs `orders_90d` permite predecir churn.

---

## Product Diversity

Mide la variedad de productos comprados por usuario:

``` python
diversity_features['product_diversity_ratio'] = (
    diversity_features['unique_products'] / diversity_features['total_items']
)
```

| Estadístico | Valor |
| :---------- | ----: |
| Media       |  0.85 |
| Mediana     |  0.91 |
| Mínimo      |  0.07 |
| Máximo      |  1.00 |

![](../../assets/Práctica11/ProductDiversity.png)

> La mayoría de los clientes presentan **alta diversidad** (ratio ≈ 1),
> mientras que los ratios < 0.5 indican **recompra frecuente**.
> Estos últimos son valiosos para estrategias de fidelización.

---

## Calendar Features – Encoding cíclico

Las variables temporales (`hora`, `día de semana`, `mes`) se transformaron mediante codificación sinusoidal:

```python
orders_df['hour_sin'] = np.sin(2 * π * hour / 24)
orders_df['dow_sin'] = np.sin(2 * π * dow / 7)
orders_df['month_sin'] = np.sin(2 * π * (month-1) / 12)
```

![](../../assets/Práctica11/encoding.png)

<div class="cards-grid media">

  <div class="card">
    <alt="Encoding cíclico" loading="lazy">
    <div class="caption">
      Representación sin/cos de variables cíclicas.  
      <small>Evita discontinuidades (23h ≈ 0h, domingo ≈ lunes).</small>
    </div>
  </div>

</div>

!!! success "Resultados:"
    El *calendar encoding* preserva la continuidad temporal, mejorando el rendimiento de modelos lineales o de distancia.
    Además, se añadieron indicadores binarios de **feriado**, **inicio/fin de mes** y **weekend**,
    que mostraron un ligero aumento en el tamaño promedio del carrito durante fines de semana.

---

## External Features – Indicadores económicos simulados

Se generaron tres variables externas mensuales:

| Variable              | Distribución simulada | Interpretación                        |
| :-------------------- | :-------------------- | :------------------------------------ |
| `gdp_growth`          | media 2.5 %, sd 0.5   | Representa crecimiento macroeconómico |
| `unemployment_rate`   | media 4.0 %, sd 0.3   | Contexto laboral                      |
| `consumer_confidence` | media 100, sd 5       | Propensión al gasto                   |

> Estas variables permiten evaluar cómo factores externos
> pueden modular la actividad de compra a lo largo del tiempo.

---

## Conclusiones generales

* La **ingeniería temporal** enriquece el dataset con señales dinámicas, permitiendo modelos más contextuales y precisos.
* `.groupby()` + `.shift()` garantizan independencia temporal, evitando *data leakage*.
* Las *rolling* y *expanding windows* complementan horizontes de corto y largo plazo.
* Las métricas **RFM** y **product diversity** ofrecen perspectivas conductuales del cliente.
* El **calendar encoding** introduce estacionalidad interpretable.
* La combinación de estas técnicas sienta las bases para **modelos predictivos de retención o demanda** en entornos de e-commerce.

!!! quote "Reflexión final"
    La práctica demuestra que el tiempo es una dimensión estructural del comportamiento de los datos.
    Integrar memoria (lags), contexto (rolling/expanding) y calendario (sin/cos) transforma datos transaccionales estáticos en series dinámicas con capacidad predictiva y explicativa.

---

## Evidencias

- [**Notebook (.ipynb)**](../../evidencias/Aurrecochea-Práctica11.ipynb)

---

## Próximos pasos (Bonus)

1. Implementar un **modelo de predicción temporal (LSTM / XGBoost temporal)** sobre las features generadas.
2. Incorporar **detección de anomalías** por usuario mediante rolling std.
3. Agregar *shocks exógenos* (promociones, feriados) como variables dummy.
4. Comparar estrategias de *feature lagging* vs *window aggregation* en validación temporal.
5. Publicar un dashboard interactivo en **Power BI** o **Streamlit** visualizando evolución por cliente.

---


## Bonus

!!! abstract "Resumen ejecutivo"
    En esta extensión *bonus* de la Práctica 11 se implementa un **modelo temporal de predicción y detección de anomalías**, orientado al análisis de comportamiento por cliente a lo largo del tiempo.  
    A partir del dataset *Daily Female Births (1959)* se simulan múltiples usuarios y se genera un pipeline de **ingeniería temporal avanzada**, que incluye *lags*, *rolling windows*, *shocks exógenos* y comparación entre estrategias de *feature aggregation*.  
    La práctica se mantiene completamente **reproducible y portable**, utilizando únicamente `scikit-learn`, `xgboost` y `matplotlib`.


!!! note "🧠 Desarrollo técnico"
    **1. Dataset y simulación multicliente**  
    Se utilizó el dataset público `daily-total-female-births.csv`, generando una simulación de cinco clientes (`C1`–`C5`) con leves variaciones de escala para permitir análisis individualizados.  
    
    **2. Ingeniería temporal**  
    Se añadieron *features* derivadas de la fecha:  
    - `dayofweek`, `month`  
    - Lags: `lag_1`, `lag_3`, `lag_7`  
    - Ventanas móviles: `rolling_mean_7`, `rolling_std_7`  
    Estas variables capturan dependencias de corto y mediano plazo, junto a patrones estacionales.  
    
    **3. Shocks exógenos**  
    Se incluyeron variables dummy:  
    - `feriado` → San Valentín, 4 de julio y Navidad  
    - `promocion` → Días múltiplos de 15  
    Estas variables actúan como *shocks* que alteran el comportamiento normal de la serie.  
    
    **4. Detección de anomalías**  
    Mediante la desviación estándar móvil (`rolling_std_7`), se etiquetan como anómalos los valores que superan en más de 2σ la media local.  
    Este método dinámico permite identificar picos inesperados por cliente.  
    
    **5. Modelado con XGBoost temporal**  
    Se compararon dos enfoques:  
    - *Lag-based model*: sensible a cambios recientes.  
    - *Window-based model*: más estable y suavizado.  
    Ambos se validaron temporalmente mediante *train/test split* por fecha, evaluando su desempeño con RMSE.  
    
    **6. Visualización y exportación**  
    Se generaron gráficos comparativos de las predicciones y un archivo `predicciones_temporales_livianas.csv` listo para visualización interactiva en **Power BI** o **Streamlit**.

## Evidencias

- [**Script (.py)**](../../evidencias/Aurrecochea-Práctica11Bonus.ipynb)

## 📊 Resultados

![](../../assets/Práctica11/Bonus.png)

| Estrategia | RMSE aprox. | Interpretación |
|-------------|--------------|----------------|
| Lag features | ~6.3 | Captura bien patrones locales, ideal para forecasting |
| Window features | ~6.8 | Suaviza picos, útil para análisis de estabilidad |
| Anomalías detectadas | 10–15 % de puntos | Mayormente tras feriados o promociones |

!!! success "🌿 Conclusiones"
    - **Las features lag y rolling windows** permiten construir modelos temporales competitivos sin recurrir a arquitecturas neuronales, reduciendo la complejidad computacional.  
    - La **detección de anomalías por cliente** aporta valor interpretativo, señalando comportamientos atípicos asociados a eventos externos o posibles errores de registro.  
    - La inclusión de **shocks exógenos** mejora la capacidad explicativa del modelo al capturar desviaciones inducidas por promociones o fechas especiales.  
    - La comparación entre *lagging* y *window aggregation* refleja la tensión entre **sensibilidad y estabilidad**: los lags reaccionan más rápido, mientras que las ventanas suavizan.  
    - La **exportación a CSV** deja el terreno listo para un dashboard interactivo, facilitando la exploración visual del comportamiento temporal y las anomalías por cliente.

---

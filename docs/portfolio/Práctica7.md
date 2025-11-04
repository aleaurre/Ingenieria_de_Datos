---
title: "07 - Detección y Corrección de Sesgo con Fairlearn"
date: 2025-09-26
number: 7
status: "Completada"
tags: [Fairlearn, Bias Detection, Boston Housing, Titanic, Ames Housing, Ethics, Transparency, Responsible AI]
notebook: docs/evidencias/Aurrecochea-Práctica7.ipynb
drive_viz: —
dataset: "Boston Housing, Titanic, Ames Housing"
time_est: "4 h 00 m"
time_spent: "—"
---

# {{ page.meta.title }}

<span class="pill">{{ page.meta.status }}</span>
<span class="pill">#{{ page.meta.number }}</span>
{% if page.meta.tags %}{% for t in page.meta.tags %}<span class="pill">{{ t }}</span>{% endfor %}{% endif %}

!!! abstract "Resumen ejecutivo"
    **Objetivo:** aplicar técnicas de detección y corrección de sesgo en tres datasets de referencia (**Boston Housing**, **Titanic**, **Ames Housing**) utilizando métricas y algoritmos de **Fairlearn**.  
    **Scope:** combinar análisis estadístico, fairness metrics y reflexión ética en casos de regresión y clasificación.  
    **Resultado:** se consolidó un **framework ético-práctico** para decidir cuándo detectar, cuándo corregir y cuándo rechazar un modelo.  

---

## Contexto
Los sesgos en modelos de ML no siempre pueden ni deben corregirse automáticamente. Esta práctica mostró cómo abordar casos históricos (Boston), sistemáticos (Titanic) y socioeconómicos (Ames), con una mirada crítica sobre la validez técnica y ética de las decisiones.

## Objetivos
- [x] Detectar sesgos en diferentes dominios.  
- [x] Evaluar métricas de fairness (Demographic Parity, Equalized Odds).  
- [x] Implementar correcciones con Fairlearn.  
- [x] Reflexionar sobre el trade-off entre utilidad y equidad.  
- [x] Proponer un marco ético de decisión.  

---

## Desarrollo

### 1) Boston Housing — Sesgo Racial Histórico
- Variable problemática: **B (proporción afroamericana)**.  
- Correlación con `MEDV`: **0.333**.  
- Brecha detectada: **–2.4%** entre grupos.  
- **Decisión ética:** solo usar con fines educativos, nunca en producción.  
- **Lección:** los sesgos históricos deben documentarse, no corregirse automáticamente.

### 2) Titanic — Sesgo Género + Clase
- Brechas detectadas: **54.8% género**, **41.3% clase social**.  
- Modelo baseline: Accuracy = 0.673, Demographic Parity Diff = 0.113.  
- Con Fairlearn (ExponentiatedGradient): Accuracy = 0.631, DP Diff = 0.062.  
- **Trade-off:** pérdida de precisión ≈ **6.2%**, ganancia en equidad moderada.  
- **Decisión ética:** evaluar caso por caso, aceptando pequeños costos de performance.  

### 3) Ames Housing — Sesgo Geográfico y Temporal
- Brecha geográfica: **45%** entre barrios caros y baratos.  
- Brecha temporal: **28%** entre casas nuevas y antiguas.  
- **Decisión ética:** en contextos hipotecarios no usar sin mitigar, porque reproduce desigualdades.  
- **Lección:** los modelos inmobiliarios tienen alto riesgo de perpetuar inequidades estructurales.  

---

## Respuestas propias de la Práctica

### Análisis Boston Housing
- **¿Es ético usar la variable B en 2025?**  
No. Incluirla en producción perpetúa prejuicios históricos en vivienda y crédito.  

- **¿Cuándo la utilidad justifica el sesgo?**  
Nunca. La predicción no justifica discriminación.  

- **Responsabilidad profesional:**  
Educar, advertir y documentar sesgos; no reproducirlos. 

- **Alternativas éticas:**  
Eliminar B, usar proxies menos problemáticos (ej. `LSTAT`, `RM`).  

- **Decisión final:**  
**USAR SOLO PARA EDUCACIÓN — NO PARA PRODUCCIÓN.**

### Análisis del Titanic

- **¿Qué sesgos se detectaron?**  
Diferencia de supervivencia de **54.8%** entre géneros.  
Diferencia de **41.3%** entre primera y tercera clase.  

- **¿Qué pasa al aplicar Fairlearn?** 
Accuracy baja de 0.673 → 0.631.  
Demographic Parity mejora (0.113 → 0.062).  

- **¿Cuál fue el trade-off?**  
Performance loss ≈ **6.2%**.  
Fairness gain ≈ 0.051.  

- **Conclusión ética:**  
El modelo justo no siempre es mejor; se debe evaluar contexto y tolerancia al sesgo.  

### Análisis en Ames Housing
- **Brechas identificadas:**  
Geográfica: **45%** entre barrios caros/baratos.  
Temporal: **28%** entre casas nuevas/antiguas.

- **Conclusión:**  
En contextos sensibles (hipotecas), usar sin mitigar puede reforzar desigualdades.  
Es preferible detectar/documentar o incluso **rechazar** el modelo.  

---

## Reflexiones éticas críticas
- **¿Cuándo detectar > corregir?**  
  En sesgos históricos (ej. Boston Housing). 

- **¿Transparencia vs utilidad?**  
  Siempre priorizar transparencia.  

- **¿Responsabilidades éticas?**  
  Identificar y advertir sesgos no corregibles.  

- **¿Qué es mejor: modelo sesgado documentado o corregido opaco?**  
  El sesgado documentado → permite auditoría. 

- **¿Cuándo rechazar deployment?**  
  Si afecta derechos fundamentales (créditos, justicia, empleo).  

---

## Métricas / Indicadores

| Dataset | Brecha detectada | Técnica aplicada | Performance | Decisión ética |
|---------|-----------------|-----------------|-------------|----------------|
| Boston Housing | –2.4% (racial) | Solo detección | R² = 0.7112 con variable B | Uso solo educativo |
| Titanic | 54.8% género / 41.3% clase | Fairlearn ExponentiatedGradient | Acc. 0.631 (–6.2%) | Evaluar trade-off |
| Ames Housing | 45% geográfica / 28% temporal | Análisis crítico | — | Evitar en crédito |

---

## Decisiones clave (ADR-lite)
- **Detectar > Corregir** cuando el sesgo es histórico y no corregible.  
- **Detectar + Corregir** cuando el sesgo es sistemático y Fairlearn ofrece mejoras razonables.  
- **Rechazar modelo** en contextos de alto impacto socioeconómico.  
- Documentar siempre las limitaciones y comunicar el sesgo de forma transparente.  

---

## Evidencias

- [**Notebook (.ipynb)**](../../evidencias/Aurrecochea-Práctica7.ipynb) — script ejecutado con las métricas y análisis de Fairlearn.  

<div class="cards-grid media">

  <div class="card">
    <img src="../../assets/Práctica7/boxplot.png" alt="Histograma y boxplot de precios por grupo racial" loading="lazy">
    <div class="caption">
      Distribución de Precios (Boston Housing)
      <small>Comparación entre grupos de alta y baja proporción afroamericana</small>
    </div>
  </div>

</div>

---

## Reflexión final
La práctica mostró que la **ética en ML no es opcional**. Detectar, corregir o rechazar un modelo no depende solo de métricas, sino del **contexto social y ético**. Los data scientists deben asumir un rol crítico: documentar, comunicar y prevenir usos indebidos.

---

## Próximos pasos
- [ ] Probar constraints adicionales (Equalized Odds, TPR/FPR Parity).  
- [ ] Experimentar con algoritmos alternativos (`GridSearch`, `ThresholdOptimizer`).  
- [ ] Aplicar análisis interseccional (múltiples variables sensibles).  
- [ ] Construir un **repositorio de fairness playbooks** para proyectos futuros.  

---

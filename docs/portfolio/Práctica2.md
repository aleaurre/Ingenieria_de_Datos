---
title: "02 — Publicación del Portafolio (GitHub Pages + MkDocs)"
date: 2025-09-07
number: 2
status: "Completado"
tags: [GitHub Pages, MkDocs, CI/CD, Deploy, Documentación]
notebook: —
drive_viz: —
dataset: —
time_est: "1 h 30 m"
time_spent: "—"
---

# {{ page.meta.title }}

<span class="pill">{{ page.meta.status }}</span>
<span class="pill">#{{ page.meta.number }}</span>
{% if page.meta.tags %}{% for t in page.meta.tags %}<span class="pill">{{ t }}</span>{% endfor %}{% endif %}

!!! abstract "Resumen ejecutivo"
    **Objetivo:** dejar online el **portafolio** con **GitHub Pages** usando **MkDocs + Material** y un workflow de **GitHub Actions**.  
    **Resultado:** sitio público accesible, estructura mínima completada (About, links a prácticas e insights iniciales) y **pipeline de despliegue** reproducible.

**Enlaces rápidos:**  
[Consigna oficial — “Publicar tu Portafolio con GitHub Pages”](https://juanfkurucz.com/ucu-id/ut1/02-portafolio-github-pages/)

---

## Contexto
Esta práctica asegura que el portafolio esté **online**, con despliegues automáticos en cada `push` a **main**. Se parte de un **template** y se publica con **GitHub Actions**, dejando secciones mínimas visibles (About, prácticas e imágenes destacadas).

## Objetivos
- [x] Crear el repo **desde el template** (no fork).  
- [x] Activar **GitHub Pages** con **Source: GitHub Actions**.  
- [x] Completar el **contenido mínimo** (About, enlaces a prácticas, visualizaciones).  
- [x] Hacer **commit + push** y verificar el **workflow** de deploy.  
- [x] Entregar **URL pública** en WebAsignatura.

---

## Actividades (con tiempos estimados)

| Actividad                                         | Estimado | Real | Nota |
|---|---:|---:|---|
| Crear repo desde template                         | 10 m | 5 m | `Use this template` → repo público y nombre corto. |
| Activar GitHub Pages (Actions)                    | 5 m  | 5 m | Settings → Pages → **Source: GitHub Actions**. |
| Completar contenido mínimo (About, links, figs)   | 45 m | 20 m | Portada, enlaces a P1 (#01 EDA Iris), capturas destacadas. |
| Publicar cambios (commit/push)                    | 10 m | 5 m | Verificar job de **deploy** (1–3 min). |
| Entrega (WebAsignatura)                           | 5 m  | 10 m | Pegar URL pública del sitio. |

> **Totales** — Estimado: **1 h 15 m** · Real: **45 m**.

---

## Desarrollo

- **Template → Repo**: creación desde `Use this template`; repo **público** y descripción corta.  
- **Pages (Actions)**: Settings → Pages → Source = **GitHub Actions**; se verifica el **workflow** en *Actions*.  
- **Contenido mínimo**:  
  - **About**: bio breve + links (GitHub/LinkedIn).  
  - **Prácticas**: enlaces relativos a #01 (EDA) y a esta #02; 3–5 **insights** y 1–2 **figs** (pairplot/heatmap).  
  - **Backlog**: “Próximos pasos” para seguir iterando.  
- **Publicación**: `git commit -m "feat: publicar estructura inicial del portafolio"` → `git push` → comprobar URL Pages.  
- **Entrega**: enviar URL pública (no la del repo) y confirmar visibilidad.

---

## Indicadores de despliegue

| Indicador                                     | Estado / Valor |
|---|---|
| URL pública (GitHub Pages)                    | **TODO**: pega aquí tu URL |
| Workflow `deploy` en Actions                  | ✅ Éxito |
| Tiempo de build (primer deploy)               | ~1–3 min |
| Secciones mínimas visibles                    | About · Prácticas · Visualizaciones |
| Accesibilidad del sitio                       | Pública (repo **Public**) |

!!! tip "Criterios de aceptación"
    - [x] Repo creado **desde template** (no fork), **público**.  
    - [x] **Pages** activado via **GitHub Actions** y deploy verde.  
    - [x] Contenido mínimo visible (About, links, figs).  
    - [x] **URL pública** funcionando y entregada.

---

## Decisiones clave (ADR-lite)
- **Source de Pages:** **GitHub Actions** (no branch `gh-pages` manual).  
- **Estructura:** mantener `docs/` + `assets/` + `mkdocs.yml` del template.  
- **Links relativos** entre prácticas para evitar roturas al mover rutas.  
- **Sin dominio custom** por ahora; considerar más adelante.

!!! warning "Riesgos / Supuestos"
    - **404 inicial** tras el deploy → suele resolverse en 2–3 min; revisar *Actions* si persiste.  
    - **URL con `/<repo>/`** (importante en GitHub Pages de proyecto).  
    - **Leak privado**: confirmar que el repo sea **público** pero sin credenciales en commits.

---

## Reproducibilidad (local)
Requiere `python 3.11` y:

```bash
pip install -q mkdocs mkdocs-material
mkdocs serve           # servidor local en http://127.0.0.1:8000
# para publicar local→gh, basta con push; el workflow de Actions despliega solo
```
---

## Evidencias

- [Sitio público (Acerca):](https://aleaurre.github.io/ia-portfolio/acerca/)

---

!!! note "Reflexión"
    Lo más valioso fue automatizar el despliegue: pasar de “publicar a mano” a CI/CD con Actions reduce fricción y errores.
    Aprendizaje clave: en GitHub Pages de proyecto la URL incluye /<repo>/, por lo que conviene usar links relativos entre páginas para evitar roturas si cambio el nombre del repo o la ruta base.
    También confirmé que Material para MkDocs resuelve rápido la navegación con tarjetas y pills, manteniendo consistencia visual con mínimas líneas en extra.css.
    A futuro, este setup permite iterar contenido sin tocar configuración: cada push publica, lo que alinea bien el ritmo de trabajo con la entrega en WebAsignatura.

---

## Referencias
-  Consigna oficial de la práctica: https://juanfkurucz.com/ucu-id/ut1/02-portafolio-github-pages/
-  MkDocs — documentación: https://www.mkdocs.org/
-  Material for MkDocs — documentación: https://squidfunk.github.io/mkdocs-material/
-  GitHub Pages — guía: https://docs.github.com/pages
-  GitHub Actions — conceptos y workflows: https://docs.github.com/actions



# Portfolio Template

Este repositorio es un **template** para documentar el **portafolio** del curso usando **MkDocs + Material** con despliegue automático a GitHub Pages.

## Cómo usar
1. Escribe únicamente en `docs/`.
2. Crea entradas en `docs/portfolio/` siguiendo `plantilla.md`.
3. Mantén el **frontmatter** en cada `.md`:
   ```yaml
   ---
   title: "Título de la página"
   date: YYYY-MM-DD
   ---
   ```
4. Usa nombres de archivo con orden: `01-titulo.md`, `02-otro.md`.
5. Enlaza recursos con rutas relativas.

## Ejecutar localmente
```bash
pip install -r requirements.txt
mkdocs serve
```

## Despliegue
Cada `push` a `main` ejecuta el build con `--strict` y publica en GitHub Pages.

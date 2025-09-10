# Tarea 3: EDA Netflix con Visualizaciones - Fill in the Blanks

> _Repositorio de la **Práctica 3**. Este proyecto contiene el notebook `Aurrecochea-Practica3.ipynb` con el desarrollo paso a paso.

## 🚀 Objetivo
Explicar y ejecutar los análisis/experimentos de la práctica de forma **reproducible**, dejando un flujo claro desde la instalación hasta la ejecución.

## 📦 Estructura
```
.
├── Aurrecochea-Practica3.ipynb
├── README.md
└── requirements.txt
```

## 🛠️ Requisitos
- Python 3.10+ (recomendado 3.11)

### Crear entorno y instalar dependencias
```bash
# 1) Crear y activar un entorno virtual (Windows PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate

# 2) Actualizar pip y wheel
python -m pip install --upgrade pip wheel

# 3) Instalar dependencias
pip install -r requirements.txt
```

## Datos
- Dataset de Netflix, Kaggele


## ✅ Resultados esperados
- Gráficos/tablas generados por el notebook.
- Métricas/insights de la práctica.

## 🔍 Reproducibilidad y buenas prácticas
- Usar **semillas aleatorias** si hay modelos/experimentos estocásticos.
- Mantener las rutas de datos **relativas** al repositorio.
- Evitar hardcodear credenciales o data sensible.

## 🧭 Siguientes pasos (idea)
- Separar lógica reusable en módulos (`src/`).
- Añadir tests mínimos (p. ej. `pytest`) para validar helpers.
- Generar un **informe** breve (`report.md` o PDF) con los hallazgos.

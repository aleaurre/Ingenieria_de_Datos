
# Práctica 1 – Análisis Exploratorio del Dataset Iris 

**Autores:** Alexia Aurrecochea & Valentín Rodríguez  
**Fecha:** 13/08/2025  
**Entorno:** Google Colab + Python 3

## Objetivo
El propósito de este notebook es realizar una **exploración inicial del dataset Iris**, disponible públicamente en el [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/53/iris). A través del uso de herramientas de visualización, manipulación y descripción estadística, se busca:
- Comprender las características principales del dataset.
- Identificar patrones, relaciones y posibles outliers.
- Organizar los outputs en una estructura reutilizable de carpetas (`results/`).

## Requisitos y dependencias
Este notebook se ejecuta en **Google Colab**, por lo que la mayoría de los paquetes ya están disponibles por defecto. 
Sin embargo, se instala `seaborn` si no está presente.

### Librerías utilizadas
- `pandas`
- `matplotlib`
- `seaborn`
- `pathlib`

### Instrucciones de instalación (en Colab)
```python
!pip install seaborn  # Solo si no está instalado
```

## Estructura del proyecto
El notebook define una estructura clara de carpetas dentro del directorio `results/`, con subdirectorios:
```
results/
├── visualizaciones/   # Gráficos generados (histogramas, pairplots, etc.)
├── perfiles/          # Archivos con perfiles estadísticos u otras salidas numéricas
└── reportes/          # Reportes textuales o exportaciones si se generan
```

## Dataset Iris
- **Tamaño:** 150 filas × 5 columnas
- **Columnas:**
  - `sepal_length` (cm)
  - `sepal_width` (cm)
  - `petal_length` (cm)
  - `petal_width` (cm)
  - `species` (Setosa, Versicolor, Virginica)

- **Fuente original:** UCI Machine Learning Repository
- **Naturaleza del problema:** Clasificación multiclase
- **Instancias**: 50 de cada Especie

## Actividades realizadas
- Montaje de Google Drive (opcional, para guardar resultados)
- Lectura y descripción del dataset
- Verificación del entorno (versiones de librerías, paths)
- Análisis visual con seaborn (distribuciones, pairplots, etc.)
- Preparación de carpetas para almacenar resultados de forma organizada

## Salidas esperadas
Al ejecutar el notebook se generan:
- Gráficos en `results/visualizaciones/`
- Archivos numéricos o perfiles si se configuran, en `results/perfiles/`
- (Opcional) Reportes descriptivos en `results/reportes/`

## Ejecución
Este notebook está pensado para ejecutarse en **Google Colab**, por lo que no requiere instalación local. 
Simplemente abrí el notebook y seguí las celdas secuencialmente.

## Notas finales
- Este trabajo es parte de una serie de prácticas de introducción al análisis exploratorio de datos.
- El formato y desarollo del README.md fue dirigido por ejemplos externos (consultar docente).
- El enfoque está en la claridad del proceso y la organización del proyecto desde el principio.
- Se prioriza reproducibilidad y orden para futuras prácticas más avanzadas (modelado, clasificación, etc).

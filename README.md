# Proyecto: Análisis de Trastornos del Sueño con Machine Learning

Este proyecto tiene como objetivo analizar datos relacionados con la salud del sueño y el estilo de vida para predecir posibles trastornos del sueño utilizando modelos de Machine Learning. 

## Estructura del Proyecto

- `Sleep&healthpython-1.ipynb`: Notebook con el análisis exploratorio de datos, preprocesamiento, y modelado (Random Forest y Ridge).
- `Sleep_health_and_lifestyle_dataset.csv`: Dataset original.
- `sleep_disorder_rf_model.pkl`: Modelo Random Forest serializado.
- `app.py`: Aplicación interactiva construida con Dash.
- `cargar_postgres.py`: Script para cargar los datos en PostgreSQL.
- `consultas_postgres.py`: Script con consultas SQL útiles al dataset.
- `Dockerfile` y `docker-compose.yml`: Para contenedorización y despliegue de la aplicación.

## Funcionalidades

- Análisis Exploratorio de Datos (EDA)
- Visualización de distribuciones, correlaciones, y estadísticas agrupadas
- Predicción de trastornos del sueño usando Random Forest y Ridge
- Evaluación del rendimiento del modelo y comparación
- Dashboard interactivo para explorar los datos y resultados

## Requisitos

- Python 3.8+
- PostgreSQL
- Bibliotecas principales: pandas, scikit-learn, imbalanced-learn, dash, plotly, seaborn, matplotlib, SQLAlchemy, psycopg2

## Instrucciones Rápidas

```bash
# 1. Clonar el repositorio
# 2. Construir la imagen y levantar el contenedor
$ docker-compose up --build

# 3. Acceder al dashboard en http://localhost:8050
```

## Autores
- María Camila Vargas
- Eliana Fuentes
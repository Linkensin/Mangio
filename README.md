# Datathon UPC – Predicción de Demanda Semanal

## Descripción del Proyecto
Este proyecto tiene como objetivo predecir la **demanda semanal de productos** utilizando un conjunto de datos proporcionado por el Datathon UPC.  
El enfoque principal combina técnicas de **ingeniería de características**, modelos de **Gradient Boosting (LightGBM)** y optimización de hiperparámetros mediante **Optuna**, incluyendo un **ensemble de modelos quantile** para capturar la variabilidad de la demanda.

El flujo de trabajo contempla:

1. Carga y limpieza de datos.  
2. Ingeniería de características (categorías, fechas, colores, embeddings de imágenes, precios contextualizados).  
3. Eliminación de ruido de características irrelevantes.  
4. Optimización de hiperparámetros con validación cruzada temporal.  
5. Entrenamiento de un ensemble de LightGBM con diferentes quantiles (`alphas`).  
6. Generación de predicciones finales para el set de test y archivo de sumisión.

---

## Estructura de Archivos

project/
├── data/
│   ├── train.csv              # Datos de entrenamiento (separador ';')
│   ├── test.csv               # Datos de prueba (separador ';')
│   └── sample_submission.csv  # Ejemplo de archivo de sumisión (separador ',')
├── notebooks/
│   └── datathon_upc_notebook.ipynb  # Notebook principal
├── submission.csv             # Archivo generado con las predicciones finales
├── README.md                  # Este archivo
└── requirements.txt           # Dependencias del proyecto

---

## Librerías Utilizadas

- **Pandas, NumPy**: Manipulación y análisis de datos.  
- **LightGBM**: Modelo de Gradient Boosting eficiente para predicción de regresión y quantile regression.  
- **Optuna**: Optimización de hiperparámetros mediante búsqueda automática y validación cruzada.  
- **Scikit-learn**: Validación cruzada temporal (`TimeSeriesSplit`), métricas y preprocesamiento.  
- **Warnings, Re, etc.**: Utilidades para limpieza de datos y manejo de errores.

---

## Flujo de Trabajo

### 1. Carga y preparación de datos
- Se cargan `train.csv` y `test.csv` con delimitador `;`.  
- Se crea la variable objetivo `total_demand` como suma de la demanda semanal por `ID`.  
- Se preparan las características estáticas de los productos, evitando **fugas de datos**.  
- El set de test se ajusta para tener las mismas columnas que el entrenamiento.

### 2. Ingeniería de Características
- **Image embeddings**: Se expanden los embeddings de 512 dimensiones en columnas separadas.  
- **Columnas categóricas y fechas**: Factorización y extracción de variables temporales (`month`, `dayofyear`).  
- **RGB de colores**: Se separa en componentes R, G, B.  
- **Precio contextual**: Se calcula el precio relativo al promedio de su familia (`price_vs_family_avg`).  
- **Limpieza de NaNs y nombres de columnas**: Para compatibilidad con LightGBM.

### 3. Eliminación de ruido
- Se eliminan **features irrelevantes** basadas en importancia 0 o embeddings poco informativos.  
- Se actualiza la lista de variables categóricas para LightGBM.

### 4. Optimización de Hiperparámetros
- Se utiliza **Optuna** con `TimeSeriesSplit(n_splits=3)` para simular la validación temporal.  
- Se optimizan parámetros de LightGBM como `learning_rate`, `num_leaves`, `feature_fraction`, `bagging_fraction`, `lambda_l1`, `lambda_l2`, y el quantile `alpha`.  
- La métrica utilizada es **Quantile Loss**, adaptada a la predicción de demanda.

### 5. Entrenamiento Final – Ensemble de Quantiles
- Se entrenan 3 modelos con alphas: `[0.7, 0.8, 0.9]`.  
- Se promedian las predicciones para generar el resultado final (**ensemble**).  
- Se realiza **clipping** para evitar valores negativos y se redondea para entregar enteros.

### 6. Generación de Sumisión
- El archivo `submission.csv` contiene:  
  - `ID`: identificador del producto.  
  - `Production`: predicción final de demanda semanal.

---

## Ejecución

1. Instalar dependencias:

```bash
pip install -r requirements.txt

# examen_nivel_intermedio_yael_adair_jimenez
# Proyecto de Funciones de Python

## Ejercicios

### Ejercicio 1: Filtrar un DataFrame
La funcion `filter_dataframe` filtra un DataFrame en funcion de una columna y un valor umbral.

- **Parametros**:
  - `df`: DataFrame a filtrar.
  - `columna_df`: Nombre de la columna para aplicar el filtro.
  - `umbral`: Valor umbral para filtrar la columna.
- **Retorna**: El DataFrame filtrado.

### Ejercicio 2: Generar Datos para Regresion
La funcion `generate_regression_data` genera un conjunto de datos simulado para un problema de regresion, utilizando variables independientes y una variable dependiente con ruido.

- **Parametros**:
  - `n_muestras`: Numero de muestras a generar.
- **Retorna**: Un DataFrame con variables independientes y una Serie con la variable dependiente.

### Ejercicio 3: Entrenar Modelo de Regresion Lineal Multiple
La funcion `train_multiple_linear_regression` entrena un modelo de regresion lineal multiple utilizando `sklearn`.

- **Parametros**:
  - `X`: DataFrame con variables independientes.
  - `y`: Serie con la variable dependiente.
- **Retorna**: Un modelo entrenado de `LinearRegression`.

### Ejercicio 4: Aplanar una Lista de Listas
La funcion `flatten_list` aplana una lista de listas en una sola lista.

- **Parametros**:
  - `lista`: Lista de listas a aplanar.
- **Retorna**: Una lista que contiene todos los elementos de las sublistas en un solo nivel.

### Ejercicio 5: Agrupar y Agregar en un DataFrame
La funcion `group_and_aggregate` agrupa un DataFrame por una columna y calcula la media de otra columna.

- **Parametros**:
  - `df`: DataFrame de entrada.
  - `group_col`: Nombre de la columna para agrupar.
  - `agg_col`: Nombre de la columna para calcular la media.
- **Retorna**: Un DataFrame con los valores agrupados y la media calculada.

### Ejercicio 6: Entrenar Modelo de Clasificacion Logistica
La funcion `train_logistic_regression` entrena un modelo de regresion logistica con un conjunto de datos binarios.

- **Parametros**:
  - `X`: DataFrame con variables independientes.
  - `y`: Serie con la variable dependiente (binaria).
- **Retorna**: Un modelo entrenado de `LogisticRegression`.

### Ejercicio 7: Aplicar Funcion a una Columna en un DataFrame
La funcion `apply_function_to_column` aplica una funcion personalizada a cada valor de una columna en un DataFrame.

- **Parametros**:
  - `df`: DataFrame de entrada.
  - `column_name`: Nombre de la columna a la que se le aplicara la funcion.
  - `func`: Funcion que se aplicara a cada valor de la columna.
- **Retorna**: Un DataFrame con la columna modificada.

### Ejercicio 8: Filtrar y Elevar al Cuadrado
La funcion `filter_and_square` filtra los numeros mayores que 5 de una lista y devuelve sus cuadrados.

- **Parametros**:
  - `numbers`: Lista de numeros.
- **Retorna**: Una lista con los cuadrados de los numeros mayores que 5.

## Requisitos

- `pandas`
- `numpy`
- `Faker`
- `scikit-learn`

Puedes instalar todo con:

```bash
pip install pandas numpy faker scikit-learn

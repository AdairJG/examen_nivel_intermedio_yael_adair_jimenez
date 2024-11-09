import pandas as pd
import numpy as np
from faker import Faker
from sklearn.linear_model import LinearRegression,LogisticRegression


####------------------- Ejercicio 1 ------------------------#######
def filter_dataframe(df,columna_df,umbral):
    """
    Filtra un DataFrame en función de una columna y un umbral

    Parametros:
    - df : El DataFrame a filtrar
    - columna_df : El nombre de la columna en la cual se aplicara el filtro
    - umbral : El valor umbral para filtrar la columna

    Retorna:
    - df_filtrado: El DataFrame filtrado
    """
    # Filtra el DataFrame
    df_filtrado = df[df[columna_df]>=umbral]
    # Retorna el DataFrame filtrado
    return df_filtrado

####------------------- Ejercicio 2 ------------------------#######

def generate_regression_data(n_muestras):
    """
    Genera un conjunto de datos simulado para un problema de regresion

    Parametros:
    - n_muestras: int, el numero de muestras que queremos generar

    Retorna:
    - Un DataFrame con las variables independientes y una Serie con la variable dependiente
    """
    fake = Faker()
    Faker.seed(0)  ## Semilla en cero para que siempre sean los mismos valores
    np.random.seed(0)

    # Generar variables independientes aleatorias
    data = {
        'feature_1': [fake.random_number(digits=2) for _ in range(n_muestras)],
        'feature_2': [fake.random_number(digits=2) for _ in range(n_muestras)],
        'feature_3': [fake.random_number(digits=2) for _ in range(n_muestras)]
    }
    # Generando el Dataframe de las varaibles
    X = pd.DataFrame(data)

    # Generar variable dependiente con algo de aleatoriedad
    y = X['feature_1'] * 0.3 + X['feature_2'] * 0.5 + X['feature_3'] * 0.2 + np.random.normal(0, 100, n_muestras)
    y = pd.Series(y)

    return X, y
####------------------- Ejercicio 3 ------------------------#######

def train_multiple_linear_regression(X, y):
    """
    Entrena un modelo de regresión lineal multiple utilizando los datos simulados

    Parametros:
    - X: DataFrame con las variables independientes
    - y: Serie con la variable dependiente

    Retorna:
    - Un modelo entrenado de LinearRegression con SK-learn
    """
    # Inicializar el modelo de regresion lineal
    modelo = LinearRegression()
    # Entrenar el modelo con los datos
    modelo.fit(X, y)
    return modelo

####------------------- Ejercicio 4 ------------------------#######

def flatten_list(lista):
    """
    Aplana una lista de listas en una sola lista

    Parametros:
    - lista: Una lista de listas

    Retorna:
    - Una nueva lista que contiene todos los elementos separados de la lista de listas
    """
    # Itera en cada sublista y crea una nueva lista aplanada
    nueva_lista = [i for sublista in lista for i in sublista]
    # Retorna la lista aplastada
    return nueva_lista


####------------------- Ejercicio 5 ------------------------#######

def group_and_aggregate(df, group_col, agg_col):
    """
    Agrupa un DataFrame por una columna y calcula la media de otra columna

    Parametros:
    - df: DataFrame de entrada
    - group_col: Nombre de la columna para agrupar
    - agg_col: Nombre de la columna para calcular la media

    Retorna:
    - DataFrame con los valores agrupados y la media calculada
    """
    # Agrupa por la columna especificada y calcula la media
    result = df.groupby(group_col)[agg_col].mean().reset_index()
    return result


####------------------- Ejercicio 6 ------------------------#######
def train_logistic_regression(X, y):
    """
    Entrena un modelo de regresión logistica utilizando un conjunto de datos binarios

    Parametros:
    - X: DataFrame con las variables independientes
    - y: Serie con la variable dependiente (binaria)

    Retorna:
    - Un modelo entrenado de LogisticRegression
    """
    # Inicializar el modelo de regresion logistica
    model = LogisticRegression()
    # Entrenar el modelo con los datos
    model.fit(X, y)

    return model


####------------------- Ejercicio 7 ------------------------#######

def apply_function_to_column(df, column_name, func):
    """
    Aplica una función personalizada a cada valor de una columna en un DataFrame

    Parametros:
    - df: DataFrame de entrada
    - column_name: Nmbre de la columna a la que se le aplicara la funcion
    - func: funcion que se aplicara a cada valor de la columna

    Retorna:
    - DataFrame con la columna modificada
    """
    # Aplica la función a la columna especificada
    df[column_name] = df[column_name].apply(func)
    return df


####------------------- Ejercicio 8 ------------------------#######

def filter_and_square(numbers):
    """
    Filtra los numeros mayores que 5 de una lista y devuelve sus cuadrados

    Parametros:
    - numbers: lista de numeros

    Retorna:
    - Lista de los cuadrados de los numeros mayores que 5
    """
    return [x ** 2 for x in numbers if x > 5]


if __name__ == "__main__":
    # ===== Ejemplo 1: Prueba de filter_dataframe =====
    df = pd.DataFrame([['Oscarito', 15], ['Maria', 45], ['Pedrito', 27]], columns=['Nombre', 'Edad'])
    print("Resultado de filter_dataframe:")
    print(filter_dataframe(df, 'Edad', 28))

    # ===== Ejemplo 2: Prueba de generate_regression_data =====
    X, y = generate_regression_data(100)
    print("\nResultado de generate_regression_data:\n")
    print(X.head())
    print(y.head())

    # ===== Ejemplo 3: Prueba de generate_regression_data =====
    # Generar los datos
    print("\nResultado de train_multiple_linear_regression:")
    X, y = generate_regression_data(100)

        # Entrenar el modelo
    modelo_entrenado = train_multiple_linear_regression(X, y)
    print("Coeficientes del modelo:", modelo_entrenado.coef_)
    print("Intercepto del modelo:", modelo_entrenado.intercept_)

    # ===== Ejemplo 4: Prueba de flatten_list =====
    print("Resultado de flatten_list:")
    A = [[1, 2, 3], [4, 5], ['Messi', 'Cristiano', 'Kaka']]
    print(flatten_list(A))

    # ===== Resultado 5: Prueba de group_and_aggregate =====
    print("\nPrueba de group_and_aggregate:")
    data = {
        'grupo': ['A', 'A', 'B', 'B', 'C', 'C'],
        'valor': [10, 20, 10, 30, 40, 50]
    }
    df = pd.DataFrame(data)
    resultado = group_and_aggregate(df, 'grupo', 'valor')
    print(resultado)

    # ===== Ejemplo 6: Prueba de train_logistic_regression =====
    print("\nResultado de train_logistic_regression:")
    np.random.seed(0)
    X = pd.DataFrame({
        'feature_1': np.random.rand(100),
        'feature_2': np.random.rand(100),
        'feature_3': np.random.rand(100)
    })
    y = pd.Series(np.random.choice([0, 1], size=100))
    modelo_entrenado = train_logistic_regression(X, y)
    print("Coeficientes del modelo:", modelo_entrenado.coef_)
    print("Intercepto del modelo:", modelo_entrenado.intercept_)

    # ===== Ejemplo 7: Prueba de apply_function_to_column =====
    print("\nResultado de apply_function_to_column:")
    data = {'valores': [1, 2, 3, 4, 5]}
    df = pd.DataFrame(data)
    def cuadrado(x):
        return x ** 2
    resultado = apply_function_to_column(df, 'valores', cuadrado)
    print(resultado)

    # ===== Ejemplo 8: Prueba de filter_and_square =====
    print("\nResultado de filter_and_square:")
    numeros = [1, 4, 6, 8, 10]
    resultado = filter_and_square(numeros)
    print(resultado)
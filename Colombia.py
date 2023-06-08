#PrediccionesCOlombia



import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Lasso

# Leer el archivo de Excel
data = pd.read_excel('PrediccionesCOlombia.xlsx')

# Obtener los valores de entrada (X) y salida (y)
X = data.iloc[:, :-5].values  # Excluir las últimas 5 columnas, que son los períodos a pronosticar
y = data.iloc[:, -5:].values  # Tomar las últimas 5 columnas como las salidas a pronosticar

# Crear el transformador polinomial
poly = PolynomialFeatures(degree=3)  # Ajusta el grado del polinomio según tus necesidades
X_poly = poly.fit_transform(X)  # Aplicar transformaciones polinomiales a los datos de entrada

# Crear el modelo de regresión polinomial
model = Lasso(alpha=0.1)  # Ajusta el valor de alpha según tus necesidades

# Crear una lista para almacenar los resultados
resultados = []

# Iterar sobre cada fila y realizar el pronóstico
for i in range(len(X)):
    # Obtener los datos de entrada y salida para la fila actual
    X_row = X_poly[i].reshape(1, -1)
    y_row = y[i].reshape(1, -1)

    # Entrenar el modelo
    model.fit(X_row, y_row)

    # Realizar el pronóstico para los próximos 5 periodos
    pronostico = model.predict(X_row)[:, -6:]

    # Agregar los resultados a la lista
    resultados.append(pronostico[0])

# Convertir la lista de resultados en un DataFrame de pandas
df_resultados = pd.DataFrame(resultados, columns=['Periodo 1', 'Periodo 2', 'Periodo 3', 'Periodo 4', 'Periodo 5', 'Periodo 6'])

# Guardar los resultados en un archivo Excel
df_resultados.to_excel('resultados.xlsx', index=False)

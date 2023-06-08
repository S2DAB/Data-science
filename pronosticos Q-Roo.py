import numpy as np
from sklearn.linear_model import LinearRegression

# Datos de entrada
x = [1.2,1.6,1.7,1.79,1.8]  # Valores de los períodos anteriores


# Crear matriz de características
X = np.array(list(range(1, len(x) + 1))).reshape((-1, 1))

# Crear variable objetivo
y = np.array(x)

# Crear modelo de regresión lineal y entrenarlo
model = LinearRegression()
model.fit(X, y)

# Generar valores para los próximos 5 períodos
proximos_periodos = np.array(list(range(len(x) + 1, len(x) + 6))).reshape((-1, 1))

# Realizar predicciones para los próximos 5 períodos
predicciones = model.predict(proximos_periodos)

print("Predicciones para los próximos 5 períodos:")
for i, prediccion in enumerate(predicciones):
    periodo = len(x) + i + 1
    print(f"Periodo {periodo}: {prediccion}")
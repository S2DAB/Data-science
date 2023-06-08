import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# cargar datos históricos de precios en un DataFrame
df = pd.read_excel('datos_lote.xlsx', index_col='fecha', parse_dates=True)
# graficar la serie de tiempo de los precios
plt.plot(df)
plt.title('Precio del lote')
plt.xlabel('Fecha')
plt.ylabel('Precio')
plt.show()
# eliminar valores atípicos
df = df[(np.abs(df - df.mean()) <= (3 * df.std())).all(axis=1)]

# interpolación de datos faltantes
df = df.interpolate(method='time')
# modelo de suavizado exponencial
modelo_exp = ExponentialSmoothing(df, trend='add', seasonal='add', seasonal_periods=12)
resultados_exp = modelo_exp.fit()

# modelo ARIMA
modelo_arima = ARIMA(df, order=(1, 1, 1))
resultados_arima = modelo_arima.fit()
# validación cruzada para modelo ARIMA
train_size = int(len(df) * 0.8)
train, test = df[:train_size], df[train_size:]
history = [x for x in train.values]
predictions = list()
for i in range(len(test)):
    modelo_arima = ARIMA(history, order=(1, 1, 1))
    modelo_arima_fit = modelo_arima.fit()
    output = modelo_arima_fit.forecast()
    yhat = output[0]
    predictions.append(yhat)
    obs = test.values[i]
    history.append(obs)
error = mean_squared_error(test, predictions)
print('Error de validación cruzada: %.3f' % error)

# análisis de residuos para modelo de suavizado exponencial
residuos = resultados_exp.resid
fig, ax = plt.subplots(2, 1)
ax[0].plot(residuos)
ax[0].set_title('Residuos')
ax[1].hist(residuos, bins=20)
ax[1].set_title('Histograma de residuos')
plt.show()
# modelo de suavizado exponencial
pronostico_exp = resultados_exp.forecast(60)

# modelo ARIMA
history = [x for x in df.values]
predictions = list()
for i in range(60):
    modelo_arima = ARIMA(history, order=(1, 1, 1))
    modelo_arima_fit = modelo_arima.fit()
    output = modelo_arima_fit.forecast()
    yhat = output[0]
    predictions.append(yhat)
    history.append(yhat)
pronostico_arima = np.array(predictions)

# graficar pronósticos
plt.plot(df)
plt.plot(pronostico_exp, label='Suavizado exponencial')
plt.plot(pronostico_arima, label='ARIMA')
plt.title('Pronóstico de precios del lote')
plt.xlabel('Fecha')
plt.ylabel('Precio')
plt.legend()
plt.show()

#/ Este código realiza un análisis de series temporales en un conjunto de datos históricos de precios de un lote determinado. Hace lo siguiente:
#Importa las bibliotecas necesarias: pandas, numpy, matplotlib.pyplot, statsmodels.tsa.arima.model.ARIMA y statsmodels.tsa.holtwinters.ExponentialSmoothing.
#Carga los datos históricos de precios en un DataFrame de Pandas.
#Grafica la serie de tiempo de los precios.
##Realiza interpolación de datos faltantes utilizando el método 'time'.
#Ajusta un modelo de suavizado exponencial a los datos utilizando la clase ExponentialSmoothing de la biblioteca statsmodels.
#Ajusta un modelo ARIMA a los datos utilizando la clase ARIMA de la biblioteca statsmodels.
#Realiza validación cruzada para el modelo ARIMA.
#Realiza un análisis de residuos para el modelo de suavizado exponencial.
#Pronostica los precios del lote para los próximos 60 períodos utilizando ambos modelos.
#Grafica los pronósticos de precios utilizando ambos modelos y los datos históricos.   /#


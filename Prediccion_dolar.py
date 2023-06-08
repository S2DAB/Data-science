import pandas as pd
from datetime import datetime, timedelta
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout

# Leer el archivo Excel
df = pd.read_excel("Datos_dolar.xlsx")

# Convertir la columna "Fecha" a formato datetime
df["Fecha"] = pd.to_datetime(df["Fecha"], format="%d-%m-%Y")

# Obtener la última fecha del archivo
ultima_fecha = df['Fecha'].max()

# Obtener los valores históricos de la columna "Valor"
valores_historicos = df['valor'].values

# Normalizar los valores históricos entre 0 y 1
scaler = MinMaxScaler(feature_range=(0, 1))
valores_historicos_normalizados = scaler.fit_transform(valores_historicos.reshape(-1, 1))

# Crear conjuntos de entrenamiento y prueba
train_size = int(len(valores_historicos_normalizados) * 0.8)
train_data = valores_historicos_normalizados[:train_size]
test_data = valores_historicos_normalizados[train_size:]

# Crear secuencias de datos para el entrenamiento y la prueba
def create_sequences(data, sequence_length):
    X = []
    y = []
    for i in range(len(data) - sequence_length):
        X.append(data[i:i+sequence_length])
        y.append(data[i+sequence_length])
    return np.array(X), np.array(y)

sequence_length = 7  # Longitud de la secuencia de datos
X_train, y_train = create_sequences(train_data, sequence_length)
X_test, y_test = create_sequences(test_data, sequence_length)

# Crear el modelo de red neuronal
model = Sequential()
model.add(LSTM(128, input_shape=(sequence_length, 1)))
model.add(Dropout(0.2))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')

# Entrenar el modelo
model.fit(X_train, y_train, epochs=200, batch_size=16, verbose=1)

# Obtener el número de días transcurridos desde la última fecha
dias_transcurridos = (ultima_fecha - df['Fecha'].min()).days

# Crear secuencia de datos para las predicciones
X_prediccion = np.array([test_data[-sequence_length:]])

# Realizar las predicciones para los datos de prueba
predicted_values = []
for _ in range(5):
    predicted_value = model.predict(X_prediccion.reshape(1, sequence_length, 1))
    predicted_values.append(predicted_value[0][0])
    X_prediccion = np.append(X_prediccion, predicted_value.reshape(1, 1, 1), axis=1)
    X_prediccion = X_prediccion[:, 1:, :]

# Desnormalizar los valores predichos
predicted_values = scaler.inverse_transform(np.array(predicted_values).reshape(-1, 1))

# Crear fechas para las predicciones
fechas_prediccion = [ultima_fecha + timedelta(days=i) for i in range(1, 6)]

# Crear un DataFrame para almacenar las fechas y las predicciones
predicciones = pd.DataFrame({'Fecha': fechas_prediccion, 'Valor_predicho': predicted_values.flatten()})

# Imprimir el DataFrame con las predicciones
print(predicciones)




import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import os
import requests
import shutil

# Ruta al archivo Excel que contiene los datos de latitud y longitud
ruta_excel = "datos.xlsx"

# Leer los datos de latitud y longitud desde el archivo Excel utilizando pandas
df = pd.read_excel(ruta_excel)

# Obtener los límites de las coordenadas
min_lat = df['latitud'].min()
max_lat = df['latitud'].max()
min_lon = df['longitud'].min()
max_lon = df['longitud'].max()

# Tasa de crecimiento de la población
tasa_crecimiento = 3.5

# Tendencia de crecimiento al norte (en grados)
tendencia_crecimiento = 0

# Directorio de salida para las imágenes generadas
directorio_salida = "mapas/"

# Crear el directorio si no existe
if not os.path.exists(directorio_salida):
    os.makedirs(directorio_salida)

# API Key de Google Maps
api_key = "TU API"
# Calcular el diámetro del círculo de crecimiento
diámetro_círculo = max(max_lat - min_lat, max_lon - min_lon) * (tasa_crecimiento / 100)

# Definir el número de pasos para el crecimiento (por ejemplo, 5 años)
num_pasos = 5
# Iterar sobre los pasos para simular el crecimiento
for i in range(num_pasos):
    # Calcular los nuevos límites del mapa con el crecimiento estimado
    min_lat -= diámetro_círculo / 2
    max_lat += diámetro_círculo / 2
    min_lon -= diámetro_círculo / 2
    max_lon += diámetro_círculo / 2

    # Generar la URL de la imagen estática de Google Maps con los nuevos límites y el círculo de crecimiento
    url = "https://maps.googleapis.com/maps/api/staticmap?center={},{}&zoom=10&size=640x480&maptype=roadmap&key={}&markers=size:mid%7Ccolor:red%7C{}&visible={}".format(
        (min_lat + max_lat) / 2, (min_lon + max_lon) / 2, api_key, df.to_string(index=False, header=False),
        df.to_string(index=False, header=False)
    )

    # Enviar la solicitud GET a la API de Google Maps y guardar la imagen en el directorio de salida
    response = requests.get(url, stream=True)
    imagen_salida = "mapas/mapa_{}.png".format(i)
    with open(imagen_salida, 'wb') as out_file:
        shutil.copyfileobj(response.raw, out_file)
    del response
# Crear el video a partir de las imágenes generadas
fps = 10  # Fotogramas por segundo
video_salida = "crecimiento_mancha_urbana.mp4"
img_size = (640, 480)  # Tamaño de las imágenes en el video

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
video = cv2.VideoWriter(video_salida, fourcc, fps, img_size)

# Leer cada imagen y agregarla al video
for i in range(num_pasos):
    img = cv2.imread("mapas/mapa_{}.png".format(i))
    img = cv2.resize(img, img_size)
    video.write(img)

# Cerrar el video y mostrar un mensaje de finalización
video.release()
print("Video generado con éxito: {}".format(video_salida))

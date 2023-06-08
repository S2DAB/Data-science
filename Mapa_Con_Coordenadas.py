import openpyxl
import requests
import folium

# Definir ubicación central del mapa
LATITUD_CENTRAL = 18.5001889
LONGITUD_CENTRAL = -88.296146

# Crear mapa con ubicación central y nivel de zoom
mapa = folium.Map(location=[LATITUD_CENTRAL, LONGITUD_CENTRAL], zoom_start=10)

# Leer archivo de Excel
workbook = openpyxl.load_workbook('prueba2.xlsx')
sheet = workbook.active

# Iterar por las filas del archivo
for row in sheet.iter_rows(min_row=2, values_only=True):
    # Obtener la descripción desde la columna "Descripción"
    descripcion = row[0]

    # Construir la URL de la API de Google Maps
    url = 'https://maps.googleapis.com/maps/api/geocode/json'
    params = {'address': descripcion, 'key': 'AIzaSyCwP-aKvC_-B2pw606TbUxvo8MWOFeTw3M'}

    # Enviar solicitud a la API de Google Maps
    response = requests.get(url, params=params)

    # Obtener la ubicación aproximada desde la respuesta
    if response.status_code == 200:
        data = response.json()
        if data['status'] == 'OK':
            location = data['results'][0]['geometry']['location']
            latitud = location['lat']
            longitud = location['lng']
            print(f"La ubicación aproximada de '{descripcion}' es ({latitud}, {longitud})")

            # Agregar un marcador en el mapa con las coordenadas de la ubicación
            folium.Marker(location=[latitud, longitud], popup=descripcion).add_to(mapa)
        else:
            print(f"No se pudo obtener la ubicación de '{descripcion}'")
    else:
        print(f"Error al enviar solicitud a la API de Google Maps para '{descripcion}'")

# Guardar mapa en un archivo HTML
mapa.save('mapa.html')


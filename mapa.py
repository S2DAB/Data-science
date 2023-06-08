import googlemaps
from datetime import datetime

gmaps = googlemaps.Client(key='TU API')
# Define la ubicación de búsqueda
location = 'Quintana Roo,México'

# Realiza una búsqueda utilizando el servicio de geocodificación inversa
results = gmaps.places('propiedades en venta ' + location)

# Imprime los resultados de la búsqueda
for result in results['results']:
    print(result['name'], result['formatted_address'])

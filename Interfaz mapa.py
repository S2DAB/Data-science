import googlemaps

# Inicializa el cliente de Google Maps con tu clave de API
gmaps = googlemaps.Client(key='AIzaSyCwP-aKvC_-B2pw606TbUxvo8MWOFeTw3M')

# Función para buscar propiedades en venta en el estado y municipio ingresados por el usuario
def buscar_propiedades():
    estado = input("Ingresa el estado: ")
    municipio = input("Ingresa el municipio: ")
    location = municipio + ", " + estado + ", México"
    query = "propiedades en venta " + location
    results = gmaps.places(query)

    # Imprime los resultados de la búsqueda
    for i, result in enumerate(results['results']):
        name = result['name']
        address = result['formatted_address']
        print(f"{name} - {address}")

# Ejemplo de uso
buscar_propiedades()

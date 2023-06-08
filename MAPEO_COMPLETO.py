import tkinter as tk
import googlemaps

# Inicializa el cliente de Google Maps con tu clave de API---
gmaps = googlemaps.Client(key='Tu API')

root = tk.Tk()
root.title("BÚSQUEDA PARA PROPIEDADES - TERRENOS/LOTES PARA LA EMPRESA M2 CAPITAL")

# Define la lista de estados
estados = ['Aguascalientes', 'BAJA CALIFORNIA', 'Baja California Sur', 'CAMPECHE', 'Chiapas', 'Chihuahua', 'Coahuila', 'Colima', 'Durango', 'GUANAJUATO', 'Guerrero', 'Hidalgo', 'Jalisco', 'Mexico City', 'Michoacan', 'Morelos', 'Nayarit', 'Nuevo Leon', 'Oaxaca', 'Puebla', 'Queretaro', 'QUINTANA ROO', 'San Luis Potosi', 'Sinaloa', 'Sonora', 'Tabasco', 'Tamaulipas', 'Tlaxcala', 'Veracruz', 'YUCATAN', 'Zacatecas']

# Define la variable para almacenar la selección del usuario de estado
selected_estado = tk.StringVar(root)

# Define la lista desplegable para que el usuario seleccione un estado
estado_dropdown = tk.OptionMenu(root, selected_estado, *estados)
estado_dropdown.pack()

# Define la variable para almacenar la selección del usuario de tipo de bienes
selected_tipo = tk.StringVar(root)
selected_tipo.set("propiedades")

# Define la lista desplegable para que el usuario seleccione el tipo de bienes que busca
tipo_dropdown = tk.OptionMenu(root, selected_tipo, "propiedades", "lotes/terrenos")
tipo_dropdown.pack()
# Función para buscar bienes raíces en venta en el estado seleccionado por el usuario
def buscar_bienes():
    location = selected_estado.get() + ", México"
    tipo_bienes = selected_tipo.get()
    query = tipo_bienes + " en venta " + location
    results = gmaps.places(query)

    # Borra cualquier resultado previo
    resultados_text.delete("1.0", tk.END)

    # Imprime los resultados de la búsqueda en la ventana
    for result in results['results']:
        resultados_text.insert(tk.END, result['name'] + " - " + result['formatted_address'] + "\n\n")
# Botón para realizar la búsqueda de bienes raíces
buscar_button = tk.Button(root, text="Buscar", command=buscar_bienes)
buscar_button.pack()

# Caja de texto para mostrar los resultados de la búsqueda
resultados_text = tk.Text(root)
resultados_text.pack()

# Función para agregar las ubicaciones encontradas a un mapa
def agregar_ubicaciones():
    # Obtiene la ubicación seleccionada por el usuario
    location = selected_estado.get() + ", México"
    tipo_bienes = selected_tipo.get()
    query = tipo_bienes + " en venta " + location
    results = gmaps.places(query)

    # Borra cualquier ubicación previa en el mapa
    mapa_widget.delete("all")

    # Centra el mapa en la ubicación seleccionada por el usuario
    geocode_result = gmaps.geocode(location)
    lat = geocode_result[0]['geometry']['location']['lat']
    lng = geocode_result[0]['geometry']['location']['lng']
    mapa_center = str(lat) + "," + str(lng)
    mapa_widget.configure(width=800, height=600)
    mapa_widget.create_rectangle(0, 0, 800, 600, fill="#ffffff")
    mapa_widget.create_text(400, 300, text="Cargando ubicaciones...", fill="#000000")

    # Obtiene las coordenadas de cada ubicación encontrada y las agrega al mapa
    for result in results['results']:
        place_id = result['place_id']
        place_details = gmaps.place(place_id)
        lat = place_details['result']['geometry']['location']['lat']
        lng = place_details['result']['geometry']['location']['lng']
        mapa_widget.create_oval(lng-5, lat-5, lng+5, lat+5, fill="#ff0000")

# Botón para agregar las ubicaciones encontradas a un mapa
mapa_button = tk.Button(root, text="Agregar a mapa", command=agregar_ubicaciones)
mapa_button.pack()

# Crea el widget del mapa
mapa_widget = tk.Canvas(root, width=500, height=500)
mapa_widget.pack()

# Función para dibujar un marcador en el mapa
def dibujar_marcador(lat, lng):
    x, y = mapa_widget.coords("mapa")
    px_per_deg_lat = 110.574 / 500
    px_per_deg_lng = 111.320 / (500 * math.cos(x / 180 * math.pi))
    px_x = (lng - y) / px_per_deg_lng
    px_y = (x - lat) / px_per_deg_lat
    mapa_widget.create_oval(px_x - 5, px_y - 5, px_x + 5, px_y + 5, fill="red")

root.mainloop()

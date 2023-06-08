import tkinter as tk
import googlemaps
import pandas as pd

# Inicializa el cliente de Google Maps con tu clave de API
gmaps = googlemaps.Client(key='TU API')

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

    # Guardar los resultados en un dataframe de pandas
    df = pd.DataFrame(columns=['Nombre', 'Dirección'])
    for result in results['results']:
        nombre = result['name']
        direccion = result['formatted_address']
        df = df.append({'Nombre': nombre, 'Dirección': direccion}, ignore_index=True)

    # Exportar el dataframe a un archivo Excel
    df.to_excel('resultados.xlsx', index=False)

    # Imprime los resultados de la búsqueda en la ventana
    for result in results['results']:
        resultados_text.insert(tk.END, result['name'] + " - " + result['formatted_address'] + "\n\n")

# Botón para realizar la búsqueda de bienes raíces
buscar_button = tk.Button(root, text="Buscar", command=buscar_bienes)
buscar_button.pack()

# Caja de texto para mostrar los resultados de la búsqueda
resultados_text = tk.Text(root)
resultados_text.pack()

root.mainloop()

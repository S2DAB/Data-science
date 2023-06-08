import folium
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
from PIL import ImageTk, Image

# Crear un objeto de mapa centrado en México
mapa = folium.Map(location=[23.6345, -102.5528], zoom_start=5)

# Agregar marcadores o polígonos al mapa si deseas
folium.Marker([19.4326, -99.1332], popup='Ciudad de México').add_to(mapa)
folium.Polygon(locations=[(32.5343, -117.0296), (25.6866, -100.3161), (20.6739, -103.4057)], color='red', fill=True).add_to(mapa)

# Guardar el mapa como archivo HTML temporal
mapa.save('mapa_mexico.html')

# Crear ventana emergente
window = tk.Tk()
window.title("Mapa de México")

# Cargar imagen del mapa en la ventana
image = Image.open('mapa_mexico.html')
photo = ImageTk.PhotoImage(image)
label = ttk.Label(window, image=photo)
label.pack()

# Eliminar archivo HTML temporal
import os
os.remove('mapa_mexico.html')

# Función para cerrar la ventana
def close_window():
    if messagebox.askokcancel("Cerrar", "¿Deseas cerrar el mapa?"):
        window.destroy()

# Agregar botón de cierre
close_button = ttk.Button(window, text="Cerrar", command=close_window)
close_button.pack()

# Mostrar ventana
window.mainloop()

import tkinter as tk
import matplotlib.pyplot as plt
from PIL import ImageTk, Image

# Precios de venta de cada nivel
precios = [3000, 20000, 50000, 100000]

# Función para calcular la ganancia máxima y la distribución de licencias
def calcular_ganancia():
    total_licencias = int(entry_total.get())

    # Calcular el número de licencias para cada nivel
    num_bronce = int(total_licencias * 0.6)
    num_plata = int((total_licencias - num_bronce) / 2)
    num_oro = int((total_licencias - num_bronce - num_plata) / 2)
    num_diamante = min(int(total_licencias * 0.05), total_licencias - num_bronce - num_plata - num_oro)

    # Calcular la ganancia máxima
    ganancia = num_bronce * precios[0] + num_plata * precios[1] + num_oro * precios[2] + num_diamante * precios[3]

    # Actualizar la etiqueta de resultado con la ganancia máxima y la distribución de licencias
    lbl_resultado.config(text=f"Ganancia máxima: {ganancia}\nDistribución de licencias: [{num_bronce}, {num_plata}, {num_oro}, {num_diamante}]")

    # Crear gráfico de barras para mostrar la distribución de licencias
    niveles = ['Bronce', 'Plata', 'Oro', 'Diamante']
    distribucion = [num_bronce, num_plata, num_oro, num_diamante]
    plt.bar(niveles, distribucion)
    plt.xlabel('Niveles')
    plt.ylabel('Licencias')
    plt.title('Distribución de Licencias')
    plt.show()

# Crear ventana
ventana = tk.Tk()
ventana.title("Asignación de Licencias")
ventana.geometry("800x600")  # Establecer el tamaño de la ventana

# Agregar una imagen
imagen = Image.open("FONDO2.jpg")  # Reemplaza "ruta_de_la_imagen.png" con la ruta de tu imagen
imagen = imagen.resize((300, 200), Image.ANTIALIAS)  # Ajustar el tamaño de la imagen
imagen = ImageTk.PhotoImage(imagen)
lbl_imagen = tk.Label(ventana, image=imagen)
lbl_imagen.pack()

# Etiqueta y campo de entrada para el número total de licencias
lbl_total = tk.Label(ventana, text="Número total de licencias:")
lbl_total.pack()
entry_total = tk.Entry(ventana)
entry_total.pack()

# Botón para calcular la ganancia máxima
btn_calcular = tk.Button(ventana, text="Calcular Ganancia", command=calcular_ganancia)
btn_calcular.pack()

# Etiqueta para mostrar el resultado
lbl_resultado = tk.Label(ventana, text="")
lbl_resultado.pack()

# Ejecutar la ventana
ventana.mainloop()

import requests

# Definir la URL base de la API del INEGI para el método de Indicadores
url_base = "https://www.inegi.org.mx/app/api/indicadores/desarrolladores/"

# Definir el ID del indicador que deseas consultar
id_indicador = "6207063404"

# Definir el idioma (por ejemplo, "es" para español)
idioma = "es"

# Definir el área geográfica (por ejemplo, "00" para México)
area_geografica = "00"

# Definir si se muestran los indicadores más recientes (por ejemplo, "true" para indicadores recientes)
recientes = "true"

# Definir la fuente de datos (por ejemplo, "BISE" para Banco de Información Socioeconómica)
fuente_datos = "BISE"

# Definir la versión de la API (por ejemplo, "2.0")
version = "2.0"

# Definir tu clave de acceso (token)
token = "3be2fdd8-b396-8d5c-6ac8-3cba8983ee9e"

# Definir el formato de respuesta (por ejemplo, "json")
formato = "json"

# Construir la URL de consulta
url_consulta = url_base + f"jsonxml/INDICATOR/{id_indicador}/{idioma}/{area_geografica}/{recientes}/{fuente_datos}/{version}/{token}?type={formato}"

# Realizar la solicitud a la API del INEGI
response = requests.get(url_consulta)

# Verificar el estado de la respuesta
if response.status_code == 200:
    # Obtener los datos en formato JSON
    datos_json = response.json()
    # Procesar y utilizar los datos según sea necesario
    print(datos_json)
else:
    print("Error al realizar la consulta. Código de estado:", response.status_code)
import pandas as pd

# Definir un diccionario con los nombres de los estados
nombres_estados = {
    "01": "Aguascalientes",
    "02": "Baja California",
    "03": "Baja California Sur",
    "04": "Campeche",
    "05": "Coahuila",
    "06": "Colima",
    "07": "Chiapas",
    "08": "Chihuahua",
    "09": "Ciudad de México",
    "10": "Durango",
    "11": "Guanajuato",
    "12": "Guerrero",
    "13": "Hidalgo",
    "14": "Jalisco",
    "15": "México",
    "16": "Michoacán",
    "17": "Morelos",
    "18": "Nayarit",
    "19": "Nuevo León",
    "20": "Oaxaca",
    "21": "Puebla",
    "22": "Querétaro",
    "23": "Quintana Roo",
    "24": "San Luis Potosí",
    "25": "Sinaloa",
    "26": "Sonora",
    "27": "Tabasco",
    "28": "Tamaulipas",
    "29": "Tlaxcala",
    "30": "Veracruz",
    "31": "Yucatán",
    "32": "Zacatecas"
}

# Obtener los datos del mapa y crear el DataFrame
datos_mapa = {
    "CVE_ENT": ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21", "22", "23", "24", "25", "26", "27", "28", "29", "30", "31", "32"],
    "Valor": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210, 220, 230, 240, 250, 260, 270, 280, 290, 300, 310, 320]
}
df_mapa = pd.DataFrame(datos_mapa)

# Agregar el nombre del estado al DataFrame
df_mapa["NombreEstado"] = df_mapa["CVE_ENT"].map(nombres_estados)

# Imprimir el DataFrame resultante
print(df_mapa)
import matplotlib.pyplot as plt

# Crear el gráfico de barras
plt.bar(df_mapa["NombreEstado"], df_mapa["Valor"])

# Personalizar el gráfico
plt.xlabel("Estado")
plt.ylabel("Valor")
plt.title("Valores por Estado")

# Rotar las etiquetas del eje x para una mejor legibilidad
plt.xticks(rotation=45)

# Mostrar el gráfico
plt.show()

import requests
from bs4 import BeautifulSoup

# URL de la página web a analizar
url = "https://remax.com.mx/propiedades/los+cabos_cabo+san+lucas_baja+california+sur_mn/residencial-terreno/venta"

# Realiza la solicitud HTTP a la página web
response = requests.get(url)

# Verifica si la solicitud fue exitosa (código de estado 200)
if response.status_code == 200:
    # Extrae el contenido HTML de la respuesta
    html_content = response.content

    # Crea un objeto BeautifulSoup para analizar el contenido HTML
    soup = BeautifulSoup(html_content, 'html.parser')

    # Encuentra todos los enlaces en la página web
    links = soup.find_all('a')

    # Filtra los enlaces que contienen palabras clave relevantes para propiedades
    palabras_clave = ['propiedad', 'inmueble', 'venta']
    enlaces_propiedades = []
    for link in links:
        href = link.get('href')
        if href is not None and any(palabra in href for palabra in palabras_clave):
            enlaces_propiedades.append(href)

    # Imprime los enlaces de las propiedades encontradas
    for enlace in enlaces_propiedades:
        print(enlace)
else:
    print("Error al cargar la página:", response.status_code)

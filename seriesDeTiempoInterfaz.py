from selenium import webdriver
from selenium.webdriver.common.by import By
import pandas as pd

url = "https://www.inmuebles24.com/terrenos-en-venta-en-tulum-hasta-5000-metros-cuadrados-cubiertos.html"

# Inicializar el controlador de Selenium para Chrome
driver = webdriver.Chrome()

# Cargar la página web en el controlador
driver.get(url)

# Esperar a que se cargue el contenido dinámico
driver.implicitly_wait(10)

# Encontrar el título de la primera propiedad en la página
title_element = driver.find_element(By.XPATH, "//div[@class='postingCard'][1]//h2[@class='postingCardTitle']")

# Extraer el texto del elemento del título
title = title_element.text

# Encontrar el precio de la primera propiedad en la página
price_element = driver.find_element(By.XPATH, "//div[@class='postingCard'][1]//span[@class='firstPrice']")

# Extraer el texto del elemento del precio
price = price_element.text

# Crear un dataframe de pandas con los datos extraídos
df = pd.DataFrame({'Título': [title], 'Precio': [price]})

# Guardar los datos en un archivo de Excel
df.to_excel('propiedades.xlsx', index=False)

# Cerrar el controlador de Selenium
driver.quit()
 lo que requiero del codigo es que me de un excel con la ubicacion,las dimenciones y lo s datos relevantes de la pagina web que estoy analizando
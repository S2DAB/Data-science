import folium

# Configura el entorno
latitud = 32.492265
longitud =-117.030952

# Longitud de Tijuana
ciudad_mapa = folium.Map(location=[latitud, longitud], zoom_start=12)

# Círculo de la mancha urbana actual
mancha_urbana_actual = folium.Circle(
    location=[latitud, longitud],
    radius=6000,  # Radio del círculo en metros
    color='blue',
    fill=True,
    fill_color='blue',
    opacity=0.6
)
mancha_urbana_actual.add_to(ciudad_mapa)

# Círculo de pronóstico de crecimiento
latitud_proyeccion = 32.495002
 # Latitud de la proyección de crecimiento
longitud_proyeccion = -117.014298  # Longitud de la proyección de crecimiento
tasa_crecimiento = 12.5  # Tasa de crecimiento (por ejemplo, 2.5%)
radio_crecimiento = 650 * (1 + tasa_crecimiento)  # Radio del círculo en metros

proyeccion_crecimiento = folium.Circle(
    location=[latitud_proyeccion, longitud_proyeccion],
    radius=radio_crecimiento,
    color='black',
    fill=True,
    fill_color='black',
    opacity=0.6
)
proyeccion_crecimiento.add_to(ciudad_mapa)

# Círculo de zonas "No óptimas"
latitud_no_optimas =32.529107 

 # Latitud de zonas "No óptimas"
longitud_no_optimas =  -116.836733
 # Longitud de zonas "No óptimas"
radio_no_optimas = 1900  # Radio del círculo en metros

zonas_no_optimas = folium.Circle(
    location=[latitud_no_optimas, longitud_no_optimas],
    radius=radio_no_optimas,
    color='red',
    fill=True,
    fill_color='red',
    opacity=0.6
)
zonas_no_optimas.add_to(ciudad_mapa)

# Círculo de zonas "Óptimas"
latitud_optimas =32.496826
# Latitud de zonas "Óptimas"
longitud_optimas = -117.113355# Longitud de zonas "Óptimas"
radio_optimas = 6000  # Radio del círculo en metros

zonas_optimas = folium.Circle(
    location=[latitud_optimas, longitud_optimas],
    radius=radio_optimas,
    color='green',
    fill=True,
    fill_color='green',
    opacity=0.6
)
zonas_optimas.add_to(ciudad_mapa)

# Añade la leyenda personalizada
from folium.plugins import FloatImage



# Define la leyenda personalizada
legend_html = '''
     <div style="position: fixed;
                 bottom: 80px; left: 80px; width: 220px; height: 160px;
                 border:2px solid grey; z-index:9999; font-size:14px;
                 background-color:white;
                 ">&nbsp; <b>Leyenda</b> <br>
                   &nbsp; Mancha Urbana Actual &nbsp; <i class="fa fa-circle"
                   style="color:blue"></i><br>
                   &nbsp; Pronóstico de Crecimiento &nbsp; <i class="fa fa-circle"
                   style="color:black></i><br>
                   &nbsp; Zona  Óptima &nbsp; <i class="fa fa-circle"
                   style="color:red"></i><br>
                   &nbsp; Zona  Óptima &nbsp; <i class="fa fa-circle"
                   style="color:green"></i><br>
                  
                   &nbsp; Zona no Óptima &nbsp; <i class="fa fa-circle"
                   style="color:red"></i><br>
     </div>
'''

# Añade la leyenda al mapa
ciudad_mapa.get_root().html.add_child(folium.Element(legend_html))

# Muestra el mapa interactivo
ciudad_mapa.save('mapa_interactivo_LosCabos.html')  # Guarda el mapa como un archivo HTML
ciudad_mapa

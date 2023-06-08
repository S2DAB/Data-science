
from pytrends.request import TrendReq
from googleapiclient.discovery import build

def obtener_tendencias_google(tema):
    pytrends = TrendReq(hl='es-ES', tz=360)
    pytrends.build_payload(kw_list=[tema])
    datos = pytrends.trending_searches()
    return datos

def obtener_tendencias_youtube(tema):
    api_key = 'TU API'
    youtube = build('youtube', 'v3', developerKey=api_key)

    request = youtube.search().list(
        part='snippet',
        maxResults=70,
        q=tema,
        type='video',
        order='viewCount'
    )
    response = request.execute()
    videos = response['items']

    tendencias = []
    for video in videos:
        titulo = video['snippet']['title']
        tendencias.append(titulo)

    return tendencias

tema = ' Bienes raices '
tendencias_google = obtener_tendencias_google(tema)
tendencias_youtube = obtener_tendencias_youtube(tema)

print(f"Tendencias de {tema} en Google Trends:")
for i, tendencia in enumerate(tendencias_google):
    print(f"{i+1}. {tendencia}")
    if i == 69:  # Mostrar solo las primeras 50 tendencias
        break

print()

print(f"Tendencias de {tema} en YouTube:")
for i, tendencia in enumerate(tendencias_youtube):
    print(f"{i+1}. {tendencia}")
    if i == 69:  # Mostrar solo las primeras 50 tendencias
        break

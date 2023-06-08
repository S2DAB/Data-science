import PyPDF2
import spacy
from spacy.tokens import Doc
from summa import summarizer

# Cargar el modelo de lenguaje en español
nlp = spacy.load("es_core_news_sm")

# Leer el archivo PDF y extraer el texto
def extract_text_from_pdf(file_path):
    # Abrir el archivo PDF
    pdf_file = open(file_path, 'rb')

    # Crear un objeto de lectura de PDF
    pdf_reader = PyPDF2.PdfReader(pdf_file)

    # Extraer el texto de todas las páginas del PDF
    text = ''
    for page in pdf_reader.pages:
        text += page.extract_text()

    # Cerrar el archivo PDF
    pdf_file.close()

    # Devolver el texto extraído
    return text

# Registrar la extensión 'polarity'
Doc.set_extension('polarity', default=None)

# Realizar análisis de texto
def perform_text_analysis(text):
    # Procesar el texto con spaCy
    doc = nlp(text)

    # Extracción de entidades
    entities = [(ent.text, ent.label_) for ent in doc.ents]

    # Detección de frases clave
    key_phrases = [phrase.text for phrase in doc.noun_chunks]

    # Análisis de sentimientos
    sentiment = doc._.polarity

    # Resumen automático
    summary = summarizer.summarize(text)

    # Imprimir los resultados
    print("Entidades:")
    print(entities)
    print("Frases clave:")
    print(key_phrases)
    print("Sentimiento:")
    print(sentiment)
    print("Resumen:")
    print(summary)

# Ruta del archivo PDF
pdf_file_path = "I-1 anexo antecedentes históricos 19012018.pdf"

# Extraer texto del archivo PDF
text = extract_text_from_pdf(pdf_file_path)

# Realizar análisis de texto
perform_text_analysis(text)

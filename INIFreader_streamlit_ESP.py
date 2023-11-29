import streamlit as st
import PyPDF2
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from googletrans import Translator
import google.generativeai as palm
from io import BytesIO

nltk.download('averaged_perceptron_tagger')

# Configurar la clave de la API de Google
palm.configure(api_key='AIzaSyCezVerubEzQc9JHz3V8hofpAlSIJXGxFQ')
models = [m for m in palm.list_models() if 'generateText' in m.supported_generation_methods]

if not models:
    st.error("No hay modelos disponibles para la generación de texto. Verifica tu configuración o vuelve a intentarlo más tarde.")
    st.stop()

model = models[0].name

# Asegúrate de haber descargado los paquetes nltk y googletrans requeridos
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('maxent_ne_chunker')
nltk.download('words')

# Configurar la clave de la API de Google Translate
translator = Translator()

def preprocess_text(text):
    stop_words = set(stopwords.words("english"))
    stemmer = PorterStemmer()
    words = word_tokenize(text.lower())
    words = [stemmer.stem(word) for word in words if word.isalnum()]
    words = [word for word in words if word not in stop_words]
    return " ".join(words)

def extract_named_entities(text):
    named_entities = set()
    try:
        ne_tree = nltk.ne_chunk(nltk.pos_tag(word_tokenize(text)))
        for subtree in ne_tree:
            if isinstance(subtree, nltk.tree.Tree):
                entity = " ".join([leaf[0] for leaf in subtree.leaves()])
                named_entities.add(entity)
    except Exception as e:
        st.error(f"Error al extraer entidades nombradas: {str(e)}")
    return named_entities

def translate_text(text, target_language='en'):
    try:
        translation = translator.translate(text, dest=target_language)
        return translation.text
    except Exception as e:
        st.error(f"Error de traducción: {str(e)}")
        return f"Error de traducción: {str(e)}"

def load_pdf(file_content):
    text = ''
    try:
        pdf_reader = PyPDF2.PdfReader(BytesIO(file_content))
        num_pages = len(pdf_reader.pages)
        
        for page_num in range(num_pages):
            text += pdf_reader.pages[page_num].extract_text()
            text += '\n'  # Agregar un salto de línea entre páginas
    except Exception as e:
        st.error(f"Error al cargar el archivo PDF: {str(e)}")
    return text

def main():
    st.title('PDF Reader with Streamlit')

    # Tab for PDF reading
    st.sidebar.subheader('PDF / Text')

    # Use st.file_uploader to directly handle the file
    uploaded_file = st.sidebar.file_uploader("Cargar PDF o archivo de texto", type=["pdf", "txt"])
    if uploaded_file is not None:
        try:
            if uploaded_file.type == "application/pdf":
                file_content = uploaded_file.getvalue()
                text_data = load_pdf(file_content)
            elif uploaded_file.type == "text/plain":
                file_content = uploaded_file.getvalue()
                text_data = file_content.decode("utf-8")
            else:
                st.error("Formato de archivo no válido. Por favor, carga un archivo PDF o de texto.")
                st.stop()
        except UnicodeDecodeError:
            st.error("Error al decodificar el archivo. Asegúrate de que el archivo esté en formato de texto.")
            st.stop()

        st.subheader('Contenido del PDF / Texto')
        st.text_area("Texto", text_data, height=500)

    # Tab for the answer
    st.sidebar.subheader('Respuesta')

    # Space to enter the question
    question = st.sidebar.text_area('Ingresa tu pregunta')

    if st.sidebar.button('Obtener Respuesta'):
        question_preprocessed = preprocess_text(question)
        question_keywords = set(question_preprocessed.split())

        named_entities = extract_named_entities(question)
        question_keywords.update(named_entities)

        chatbot_input = (
            "act as a investigator expert in writing. use cohesion, semantic and a cientific style to write. please answer the next question: " + question + "." +
            " I will also provide you with information and context to solve the question and you must return the answer to the question in a paragraph briefing the information provided" +
            "this is the information you need to take into account to answer " + question + ": " + text_data
        )

        try:
            chunk_size = 5000
            chunks = [chatbot_input[i:i+chunk_size] for i in range(0, len(chatbot_input), chunk_size)]

            translated_output = ""

            for chunk in chunks:
                if not all(ord(char) < 128 for char in chunk):
                    chunk = translate_text(chunk, target_language='en')

                completion = palm.generate_text(
                    model=model,
                    prompt=chunk,
                    temperature=0,
                    max_output_tokens=800,
                )

                if completion.result is not None:
                    translated_chunk = completion.result
                    translated_output += translated_chunk

            # Traducir la respuesta completa al español antes de mostrarla
            translated_output = translate_text(translated_output, target_language='es')

            st.subheader('Respuesta')
            st.text_area("Respuesta", translated_output, height=200)

        except Exception as e:
            st.error(f"Error: {str(e)}")

if __name__ == '__main__':
    main()

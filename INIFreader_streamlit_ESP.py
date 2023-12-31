import streamlit as st
import PyPDF2
import docx
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from googletrans import Translator
import google.generativeai as palm
from io import BytesIO

nltk.download('averaged_perceptron_tagger')

# Configure Google API key
palm.configure(api_key='AIzaSyCezVerubEzQc9JHz3V8hofpAlSIJXGxFQ')  # Replace with your actual API key
models = [m for m in palm.list_models() if 'generateText' in m.supported_generation_methods]

if not models:
    st.error("No models available for text generation. Check your configuration or try again later.")
    st.stop()

model = models[0].name

# Download necessary NLTK and Googletrans packages
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('maxent_ne_chunker')
nltk.download('words')

# Configure Google Translate API key
translator = Translator()

def preprocess_text(text):
    stop_words = set(stopwords.words("english"))
    stemmer = PorterStemmer()
    words = word_tokenize(text.lower())
    words = [stemmer.stem(word) for word in words if word.isalnum()]
    words = [word for word in words if word not in stop_words]
    return " ".join(words)

def translate_text_chunked(text, target_language='en', chunk_size=5000):
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    translated_chunks = []

    for chunk in chunks:
        try:
            translation = translator.translate(chunk, dest=target_language)
            translated_chunks.append(translation.text)
        except Exception as e:
            st.error(f"Translation error: {str(e)}")
            translated_chunks.append(f"Translation error: {str(e)}")

    return ''.join(translated_chunks)

def clean_special_characters(text):
    # Remove non-English special characters
    cleaned_text = ''.join(char if ord(char) < 128 else ' ' for char in text)
    # Remove multiple spaces and newlines
    cleaned_text = ' '.join(cleaned_text.split())
    return cleaned_text

def optimize_text_data(text_data, question_keywords, context_window=100):
    # Tokenize the text for processing
    words = word_tokenize(text_data.lower())
    
    # Find indices of relevant keywords in the text
    keyword_indices = [i for i, word in enumerate(words) if word in question_keywords]

    # Include a context window around each keyword
    start_indices = [max(0, idx - context_window) for idx in keyword_indices]
    end_indices = [min(len(words), idx + context_window) for idx in keyword_indices]

    # Extract the relevant portion of the text
    relevant_text = ' '.join(' '.join(words[start:end]) for start, end in zip(start_indices, end_indices))

    return relevant_text

def generate_response(question, text_data):
    question = translate_text_chunked(question, target_language='en')  # Translate user question to English
    question_preprocessed = preprocess_text(question)
    question_keywords = set(question_preprocessed.split())

    # Optimize text_data based on question keywords
    text_data = optimize_text_data(text_data, question_keywords)

    chatbot_input = (
        "Please act as an INIF enterprises chatbot personal assistant that answers questions with natural language with an Amiable yet professional tone and always ready to respond. I'm going to provide you with information and a question. here is the question: " + question +
        " which you should respond to considering the context I give you above. You need to give me one or more paragraphs by rearranging the information I provide, attempting to answer the question, and "
        "I will also provide you with information and context to solve the question, and you must return the answer in a paragraph briefing the information provided. "
        "adding any other knowledge you have on the topic. The context information is as follows: " + text_data
    )

    try:
        # Adjust chunk size dynamically based on payload limit
        chunk_size = min(5000, len(chatbot_input))
        chunks = [chatbot_input[i:i + chunk_size] for i in range(0, len(chatbot_input), chunk_size)]

        translated_output = ""

        for chunk in chunks:
            if not all(ord(char) < 128 for char in chunk):
                chunk = translate_text_chunked(chunk, target_language='en')

            completion = palm.generate_text(
                model=model,
                prompt=chunk,
                temperature=0,
                max_output_tokens=1000,
            )

            if completion.result is not None:
                translated_chunk = completion.result
                translated_output += translated_chunk

        # Translate the complete response to English before displaying
        translated_output = translate_text_chunked(translated_output, target_language='en')

        # Combine the original question and the translated output
        combined_text = question + "\n\n" + translated_output

        # Feed the combined text back into the chatbot for a final response
        final_response = palm.generate_text(
            model=model,
            prompt=combined_text,
            temperature=0,
            max_output_tokens=1000,
        )

        # Translate the final response to Spanish before displaying
        if final_response.result is not None:
            final_response = translate_text_chunked(final_response.result, target_language='es')
            st.subheader('Response')
            st.text_area("Response", final_response, height=200)
        else:
            st.error("La pregunta que esta realizando puede que vaya en contra de las políticas de Google Bard e INIF. Por favor, reformule su pregunta sin temas no permitidos o pregunte algo diferente. Para mas informacion consulte: https://policies.google.com/terms/generative-ai/use-policy o www.inif.com.co/laura-chatbot/use-policy")

    except Exception as e:
        st.error(f"Error: {str(e)}")

def load_text(file_content, file_type):
    text = ''
    try:
        if file_type == "application/pdf":
            pdf_reader = PyPDF2.PdfReader(BytesIO(file_content))
            num_pages = len(pdf_reader.pages)

            for page_num in range(num_pages):
                text += pdf_reader.pages[page_num].extract_text()
                text += '\n'  # Add a line break between pages
        elif file_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            docx_reader = docx.Document(BytesIO(file_content))
            for paragraph in docx_reader.paragraphs:
                text += paragraph.text + '\n'
        else:
            text = file_content.decode("utf-8")
    except UnicodeDecodeError:
        st.error("Error decoding the file. Make sure the file is in text format.")
    return text

def main():
    st.title('INIFReader - Multimedia Text File Reading Tool')

    # Tab for text reading
    st.sidebar.subheader('Text')

    # Use st.file_uploader to handle the file
    uploaded_file = st.sidebar.file_uploader("Upload PDF or text file", type=["pdf", "txt", "docx"])

    if uploaded_file is not None:
        try:
            file_content = uploaded_file.getvalue()
            file_type = uploaded_file.type

            # Translate text to English in chunks
            translated_text = translate_text_chunked(load_text(file_content, file_type))

            # Save the translated text to a file
            with open("translated_text.txt", "w", encoding="utf-8") as f:
                f.write(translated_text)

            text_data = clean_special_characters(translated_text)

        except UnicodeDecodeError:
            st.error("Error decoding the file. Make sure the file is in text format.")
            st.stop()

        st.subheader('Text Content')
        st.text_area("Text", text_data, height=500)

    # Tab for the answer
    st.sidebar.subheader('Response')

    # Space to enter the question
    question = st.sidebar.text_area('Enter your question')

    if st.sidebar.button('Get Answer'):
        generate_response(question, text_data)

if __name__ == '__main__':
    main()

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

def extract_named_entities(text):
    named_entities = set()
    try:
        ne_tree = nltk.ne_chunk(nltk.pos_tag(word_tokenize(text)))
        for subtree in ne_tree:
            if isinstance(subtree, nltk.tree.Tree):
                entity = " ".join([leaf[0] for leaf in subtree.leaves()])
                named_entities.add(entity)
    except Exception as e:
        st.error(f"Error extracting named entities: {str(e)}")
    return named_entities

def translate_text(text, target_language='en'):
    try:
        translation = translator.translate(text, dest=target_language)
        return translation.text
    except Exception as e:
        st.error(f"Translation error: {str(e)}")
        return f"Translation error: {str(e)}"

def clean_special_characters(text):
    # Remove non-English special characters
    cleaned_text = ''.join(char if ord(char) < 128 else ' ' for char in text)
    return cleaned_text

def load_text(file_content, file_type):
    text = ''
    try:
        if file_type == "application/pdf":
            pdf_reader = PyPDF2.PdfReader(BytesIO(file_content))
            num_pages = len(pdf_reader.pages)
            
            for page_num in range(num_pages):
                text += pdf_reader.pages[page_num].extract_text()
                text += '\n'  # Add a line break between pages
        else:
            text = file_content.decode("utf-8")
    except UnicodeDecodeError:
        st.error("Error decoding the file. Make sure the file is in text format.")
    return text

def main():
    st.title('PDF Reader with Streamlit')

    # Tab for text reading
    st.sidebar.subheader('Text')

    # Use st.file_uploader to handle the file
    uploaded_file = st.sidebar.file_uploader("Upload PDF or text file", type=["pdf", "txt", "doc", "docx"])
    
    if uploaded_file is not None:
        try:
            file_content = uploaded_file.getvalue()
            file_type = uploaded_file.type

            text_data = clean_special_characters(load_text(file_content, file_type))
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
        question_preprocessed = preprocess_text(question)
        question_keywords = set(question_preprocessed.split())

        named_entities = extract_named_entities(question)
        question_keywords.update(named_entities)

        chatbot_input = (
            "Please act as a INIF entrerprises chatbot personal assistant that answers questions with natural language with an Amiable yet professional tone and always ready to respond. I'm going to provide you with information and a question. here is the question: " + question +
            " which you should respond to considering the context i give you above. You need to give me one or more paragraphs by rearranging the information I provide, attempting to answer the question and "
            "I will also provide you with information and context to solve the question, and you must return the answer in a paragraph briefing the information provided. "
            "adding any other knowledge you have on the topic. The context information is as follows: " + text_data
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

            # Translate the complete response to Spanish before displaying
            translated_output = translate_text(translated_output, target_language='es')

            st.subheader('Response')
            st.text_area("Response", translated_output, height=200)

        except Exception as e:
            st.error(f"Error: {str(e)}")

if __name__ == '__main__':
    main()

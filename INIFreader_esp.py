import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QTextEdit, QVBoxLayout, QPushButton, QWidget, QFileDialog, QLabel, QTabWidget
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon
import fitz
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from googletrans import Translator
import google.generativeai as palm
import time

palm.configure(api_key='YOUR_GOOGLE_API_KEY_HERE')
models = [m for m in palm.list_models() if 'generateText' in m.supported_generation_methods]

if not models:
    print("No models available for text generation. Check your configuration or try again later.")
    exit()

model = models[0].name

# Make sure you have downloaded the required nltk and googletrans packages
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('maxent_ne_chunker')
nltk.download('words')

# Set up your Google Translate API key
translator = Translator()

class PDFReaderApp(QMainWindow):
    def __init__(self):
        super().__init__()

        self.initUI()

    def initUI(self):
        self.setWindowTitle('PDF Reader')
        self.setGeometry(100, 100, 1000, 600)

        self.central_widget = QTabWidget()
        self.setCentralWidget(self.central_widget)

        # Tab for PDF reading
        pdf_tab = QWidget()
        layout_pdf = QVBoxLayout()

        # Button to load PDF or text file
        self.load_button = QPushButton('Load PDF or Text', self)
        self.load_button.clicked.connect(self.load_file)
        layout_pdf.addWidget(self.load_button)

        # Space for reading PDF or text file
        self.pdf_text = QTextEdit(self)
        self.pdf_text.setReadOnly(True)
        layout_pdf.addWidget(self.pdf_text)

        pdf_tab.setLayout(layout_pdf)

        # Tab for the answer
        answer_tab = QWidget()
        layout_answer = QVBoxLayout()

        # Space to enter the question
        self.question_label = QLabel('Enter your question:', self)
        layout_answer.addWidget(self.question_label)

        self.question_input = QTextEdit(self)
        layout_answer.addWidget(self.question_input)

        # Button to get the answer
        self.answer_button = QPushButton('Get Answer', self)
        self.answer_button.clicked.connect(self.answer_question)
        layout_answer.addWidget(self.answer_button)

        # Space to display the organized and summarized answer
        self.answer_label = QLabel('Answer:', self)
        layout_answer.addWidget(self.answer_label)

        self.answer_output = QTextEdit(self)
        self.answer_output.setReadOnly(True)
        layout_answer.addWidget(self.answer_output)

        answer_tab.setLayout(layout_answer)

        # Add tabs to the QTabWidget
        self.central_widget.addTab(pdf_tab, "PDF / Text")
        self.central_widget.addTab(answer_tab, "Answer")

        self.pdf_document = None
        self.text_data = None
        self.response_file_name = 'response.txt'

    def load_file(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, 'Load PDF or Text', '', 'Text Files (*.pdf .txt);;All Files ()', options=options)

        if file_path:
            if file_path.endswith('.pdf'):
                self.load_pdf(file_path)
            else:
                with open(file_path, 'r', encoding='utf-8') as file:
                    text = file.read()
                    self.pdf_text.setPlainText(text)
                    self.text_data = sent_tokenize(text)

    def load_pdf(self, file_path):
        self.pdf_document = fitz.open(file_path)
        text = ''
        for page in self.pdf_document.pages():
            text += page.get_text()
        self.pdf_text.setPlainText(text)
        self.text_data = sent_tokenize(text)

    def preprocess_text(self, text):
        stop_words = set(stopwords.words("english"))
        stemmer = PorterStemmer()
        words = word_tokenize(text.lower())
        words = [stemmer.stem(word) for word in words if word.isalnum()]
        words = [word for word in words if word not in stop_words]
        return " ".join(words)

    def extract_named_entities(self, text):
        ne_tree = nltk.ne_chunk(nltk.pos_tag(word_tokenize(text)))
        named_entities = set()
        for subtree in ne_tree:
            if isinstance(subtree, nltk.tree.Tree):  
                entity = " ".join([leaf[0] for leaf in subtree.leaves()])
                named_entities.add(entity)
        return named_entities

    def answer_question(self):
        if self.text_data:
            question = self.question_input.toPlainText()
            question_preprocessed = self.preprocess_text(question)
            question_keywords = set(question_preprocessed.split())

            named_entities = self.extract_named_entities(question)
            question_keywords.update(named_entities)

            chatbot_input = (
                "act as a investigator expert in writing. use cohesion, semantic and a cientific style to write. please answer the next question: " + question + "." +
                " I will also provide you with information and context to solve the question and you must return the answer to the question in a paragraph briefing the information provided" +
                "this is the information you need to take into account to answer " + question + ": " + " ".join(self.text_data)
            )

            try:
                chunk_size = 5000
                chunks = [chatbot_input[i:i+chunk_size] for i in range(0, len(chatbot_input), chunk_size)]

                translated_output = ""

                for chunk in chunks:
                    if not all(ord(char) < 128 for char in chunk):  # Check if the chunk contains non-ASCII characters
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

                with open(self.response_file_name, 'w', encoding='utf-8') as file:
                    file.write(translated_output)

                # Llamada a la nueva función para traducir y mostrar en la segunda pestaña
                self.translate_and_display()

            except Exception as e:
                self.answer_output.setPlainText(f"Error: {str(e)}")

    def translate_and_display(self):
        try:
            # Leer el contenido del archivo .txt
            with open(self.response_file_name, 'r', encoding='utf-8') as file:
                response_content = file.read()

            # Traducir el contenido al español
            translated_content = translate_text(response_content, target_language='es')

            # Guardar el contenido traducido en un segundo archivo .txt
            translated_file_name = 'response_es.txt'
            with open(translated_file_name, 'w', encoding='utf-8') as file:
                file.write(translated_content)

            # Mostrar el contenido traducido en la segunda pestaña
            self.answer_output.clear()
            self.answer_output.setPlainText(translated_content)

        except Exception as e:
            self.answer_output.setPlainText(f"Translation and Display Error: {str(e)}")

def translate_text(text, target_language='en'):
    try:
        translation = translator.translate(text, dest=target_language)
        return translation.text
    except Exception as e:
        print(f"Translation error: {str(e)}")
        return f"Translation error: {str(e)}"

def main():
    app = QApplication(sys.argv)
    app.setWindowIcon(QIcon('icon.png'))
    ex = PDFReaderApp()
    ex.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()

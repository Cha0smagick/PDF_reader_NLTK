import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QTextEdit, QVBoxLayout, QPushButton, QWidget, QFileDialog, QLabel
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon
import fitz  # PyMuPDF
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import freeGPT
import re

class PDFReaderApp(QMainWindow):
    def __init__(self):
        super().__init__()

        self.initUI()

    def initUI(self):
        self.setWindowTitle('PDF Reader')
        self.setGeometry(100, 100, 800, 600)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        layout = QVBoxLayout()

        # Botón para cargar el PDF
        self.load_button = QPushButton('Cargar PDF', self)
        self.load_button.clicked.connect(self.load_pdf)
        layout.addWidget(self.load_button)

        # Espacio de lectura del PDF
        self.pdf_text = QTextEdit(self)
        self.pdf_text.setReadOnly(True)
        layout.addWidget(self.pdf_text)

        # Espacio para ingresar la pregunta
        self.question_label = QLabel('Ingrese su pregunta:', self)
        layout.addWidget(self.question_label)

        self.question_input = QTextEdit(self)
        layout.addWidget(self.question_input)

        # Botón para obtener la respuesta
        self.answer_button = QPushButton('Obtener Respuesta', self)
        self.answer_button.clicked.connect(self.answer_question)
        layout.addWidget(self.answer_button)

        # Espacio para mostrar la respuesta organizada y resumida
        self.answer_label = QLabel('Respuesta:', self)
        layout.addWidget(self.answer_label)

        self.answer_output = QTextEdit(self)
        self.answer_output.setReadOnly(True)
        layout.addWidget(self.answer_output)

        self.central_widget.setLayout(layout)

        self.pdf_document = None
        self.text_data = None

    def load_pdf(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, 'Cargar PDF', '', 'PDF Files (*.pdf);;All Files (*)', options=options)

        if file_path:
            self.pdf_document = fitz.open(file_path)
            text = ''
            for page in self.pdf_document:
                text += page.get_text()
            self.pdf_text.setPlainText(text)
            self.text_data = sent_tokenize(text)

    def preprocess_text(self, text):
        # Tokenización, eliminación de stopwords y stemming
        stop_words = set(stopwords.words("english"))
        stemmer = PorterStemmer()
        words = word_tokenize(text.lower())
        words = [stemmer.stem(word) for word in words if word.isalnum()]
        words = [word for word in words if word not in stop_words]
        return " ".join(words)

    def answer_question(self):
        if self.pdf_document and self.text_data:
            question = self.question_input.toPlainText()
            question = self.preprocess_text(question)
            question_keywords = set(question.split())  # Palabras clave de la pregunta

            # Buscar frases y párrafos que contienen palabras clave
            response = ""
            max_output_length = 1000  # Longitud máxima del output
            current_length = 0

            for paragraph in self.text_data:
                if any(keyword in paragraph for keyword in question_keywords):
                    if current_length + len(paragraph) + 2 <= max_output_length:
                        response += paragraph + "\n\n"
                        current_length += len(paragraph) + 2
                    else:
                        break  # Romper si excede la longitud máxima

            # Llamada a FreeGPT para mejorar la respuesta
            improved_response = self.improve_response(response)

            self.answer_output.setPlainText(improved_response)

    def improve_response(self, response):
        # Llamada a FreeGPT para mejorar la respuesta
        try:
            resp = freeGPT.gpt4.Completion().create(response)
            formatted_resp = re.sub(r'\\u([0-9a-fA-F]{4})', lambda x: chr(int(x.group(1), 16)), resp)
            return formatted_resp
        except Exception as e:
            print("Error al llamar a FreeGPT:", e)
            return response

def main():
    app = QApplication(sys.argv)
    app.setWindowIcon(QIcon('icon.png'))
    ex = PDFReaderApp()
    ex.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()

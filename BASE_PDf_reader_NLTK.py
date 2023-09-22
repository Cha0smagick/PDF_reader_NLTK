import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QTextEdit, QVBoxLayout, QPushButton, QWidget, QFileDialog, QLabel
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon
import fitz  # PyMuPDF
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
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

        # Espacio para mostrar la respuesta
        self.answer_label = QLabel('Respuesta:', self)
        layout.addWidget(self.answer_label)

        self.answer_output = QTextEdit(self)
        self.answer_output.setReadOnly(True)
        layout.addWidget(self.answer_output)

        self.central_widget.setLayout(layout)

        self.pdf_document = None
        self.text_data = None
        self.preprocessed_text_data = None  # Almacena el texto del PDF preprocesado
        self.tfidf_vectorizer = None  # Almacena el vectorizador TF-IDF
        self.tfidf_matrix = None  # Almacena la matriz TF-IDF

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
            self.preprocessed_text_data = [self.preprocess_text(sentence) for sentence in self.text_data]

            # Calcula la matriz TF-IDF una sola vez
            self.tfidf_vectorizer = TfidfVectorizer()
            self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.preprocessed_text_data)

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

            # TF-IDF Vectorization de la pregunta
            question_tfidf = self.tfidf_vectorizer.transform([question])

            # Cálculo de similitud de coseno entre la pregunta y las oraciones
            cosine_similarities = cosine_similarity(question_tfidf, self.tfidf_matrix)

            # Obtención de las oraciones más similares (por ejemplo, las 3 mejores)
            num_best_sentences = 3
            most_similar_sentence_indices = cosine_similarities.argsort()[0][-num_best_sentences:][::-1]

            # Construcción de la respuesta con contexto
            response = ""
            for index in most_similar_sentence_indices:
                response += self.text_data[index] + "\n"

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

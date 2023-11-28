import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QTextEdit, QVBoxLayout, QPushButton, QWidget, QFileDialog, QLabel, QTabWidget
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon
import fitz  # Reemplaza 'PyMuPDF' con 'fitz'
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Asegúrate de haber descargado los paquetes requeridos de nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('maxent_ne_chunker')
nltk.download('words')

class PDFReaderApp(QMainWindow):
    def __init__(self):
        super().__init__()

        self.initUI()

    def initUI(self):
        self.setWindowTitle('PDF Reader')
        self.setGeometry(100, 100, 1000, 600)

        self.central_widget = QTabWidget()
        self.setCentralWidget(self.central_widget)

        # Pestaña para la lectura del PDF
        pdf_tab = QWidget()
        layout_pdf = QVBoxLayout()

        # Botón para cargar el PDF o archivo de texto
        self.load_button = QPushButton('Cargar PDF o Texto', self)
        self.load_button.clicked.connect(self.load_file)
        layout_pdf.addWidget(self.load_button)

        # Espacio de lectura del PDF o archivo de texto
        self.pdf_text = QTextEdit(self)
        self.pdf_text.setReadOnly(True)
        layout_pdf.addWidget(self.pdf_text)

        pdf_tab.setLayout(layout_pdf)

        # Pestaña para la respuesta
        answer_tab = QWidget()
        layout_answer = QVBoxLayout()

        # Espacio para ingresar la pregunta
        self.question_label = QLabel('Ingrese su pregunta:', self)
        layout_answer.addWidget(self.question_label)

        self.question_input = QTextEdit(self)
        layout_answer.addWidget(self.question_input)

        # Botón para obtener la respuesta
        self.answer_button = QPushButton('Obtener Respuesta', self)
        self.answer_button.clicked.connect(self.answer_question)
        layout_answer.addWidget(self.answer_button)

        # Espacio para mostrar la respuesta organizada y resumida
        self.answer_label = QLabel('Respuesta:', self)
        layout_answer.addWidget(self.answer_label)

        self.answer_output = QTextEdit(self)
        self.answer_output.setReadOnly(True)
        layout_answer.addWidget(self.answer_output)

        answer_tab.setLayout(layout_answer)

        # Agregar las pestañas al QTabWidget
        self.central_widget.addTab(pdf_tab, "PDF / Texto")
        self.central_widget.addTab(answer_tab, "Respuesta")

        self.pdf_document = None
        self.text_data = None

    def load_file(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, 'Cargar PDF o Texto', '', 'Archivos de Texto (*.pdf .txt);;Todos los Archivos ()', options=options)

        if file_path:
            if file_path.endswith('.pdf'):
                self.load_pdf(file_path)
            else:
                with open(file_path, 'r', encoding='utf-8') as file:
                    text = file.read()
                    self.pdf_text.setPlainText(text)
                    self.text_data = sent_tokenize(text)

    def load_pdf(self, file_path):
        self.pdf_document = fitz.open(file_path)  # Cambia 'PyMuPDF' a 'fitz'
        text = ''
        for page in self.pdf_document.pages():
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

            response = ""
            max_output_length = 1000  
            current_length = 0

            for paragraph in self.text_data:
                paragraph_preprocessed = self.preprocess_text(paragraph)
                paragraph_keywords = set(paragraph_preprocessed.split())
                common_keywords = question_keywords.intersection(paragraph_keywords)

                if common_keywords:
                    sentences = sent_tokenize(paragraph)
                    relevant_sentences = [sentence for sentence in sentences if any(keyword in sentence for keyword in question_keywords)]

                    for sentence in relevant_sentences:
                        sentence_length = len(sentence) + 2
                        if current_length + sentence_length <= max_output_length:
                            response += sentence + "\n\n"
                            current_length += sentence_length
                        else:
                            break  

            self.answer_output.setPlainText(response)

def main():
    app = QApplication(sys.argv)
    app.setWindowIcon(QIcon('icon.png'))
    ex = PDFReaderApp()
    ex.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()

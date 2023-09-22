import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QTextEdit, QVBoxLayout, QPushButton, QWidget, QFileDialog
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon
import fitz  # PyMuPDF
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

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

        self.pdf_text = QTextEdit(self)
        self.pdf_text.setReadOnly(True)
        layout.addWidget(self.pdf_text)

        self.chat_input = QTextEdit(self)
        layout.addWidget(self.chat_input)

        self.answer_output = QTextEdit(self)
        self.answer_output.setReadOnly(True)
        layout.addWidget(self.answer_output)

        self.load_button = QPushButton('Cargar PDF', self)
        self.load_button.clicked.connect(self.load_pdf)
        layout.addWidget(self.load_button)

        self.answer_button = QPushButton('Responder', self)
        self.answer_button.clicked.connect(self.answer_question)
        layout.addWidget(self.answer_button)

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
            question = self.chat_input.toPlainText()
            question = self.preprocess_text(question)

            # Preprocesamiento del texto del PDF
            preprocessed_text_data = [self.preprocess_text(sentence) for sentence in self.text_data]

            # TF-IDF Vectorization
            tfidf_vectorizer = TfidfVectorizer()
            tfidf_matrix = tfidf_vectorizer.fit_transform(preprocessed_text_data)

            # TF-IDF Vectorization de la pregunta
            question_tfidf = tfidf_vectorizer.transform([question])

            # Cálculo de similitud de coseno entre la pregunta y las oraciones
            cosine_similarities = cosine_similarity(question_tfidf, tfidf_matrix)

            # Obtención de la oración más similar
            most_similar_sentence_index = cosine_similarities.argmax()
            answer = self.text_data[most_similar_sentence_index]

            self.answer_output.setPlainText(answer)

def main():
    app = QApplication(sys.argv)
    app.setWindowIcon(QIcon('icon.png'))
    ex = PDFReaderApp()
    ex.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()

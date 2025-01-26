from flask import Flask, render_template, request
import pickle
import re
from nltk.corpus import stopwords
import os

# Ruta de modelo
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'Analisis de sentimientos', 'model2.pkl')

# Carga de modelo
with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)

# Asegurarte de descargar las stopwords si no están disponibles
try:
    stop_words = set(stopwords.words('spanish'))
except:
    import nltk
    nltk.download('stopwords')
    stop_words = set(stopwords.words('spanish'))

# Función para preprocesar texto
def preprocess_text(text):
    # Convertir a minúsculas
    text = text.lower()
    # Eliminar caracteres no alfanuméricos
    text = re.sub(r'\W+', ' ', text)
    # Quitar palabras vacías
    text = ' '.join(word for word in text.split() if word not in stop_words)
    return text

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    if request.method == 'POST':
        text = request.form['text']  # Capturar el texto ingresado por el usuario

        # Preprocesar el texto antes de pasarlo al modelo
        processed_text = preprocess_text(text)

        # Realizar predicción con el modelo
        prediction = model.predict([processed_text])  # Pasar el texto como lista
        
        # Determinar el sentimiento basado en la predicción
        if prediction[0] == 'P':
            sentiment_label = 'Positivo'
        elif prediction[0] == 'N':
            sentiment_label = 'Negativo'
        else:
            sentiment_label = 'Neutro'

        return render_template('result.html', text=text, sentiment=sentiment_label)

if __name__ == '__main__':
    app.run(debug=True)
from flask import Flask, render_template, request
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)

# Load model Naive Bayes
with open('naive_bayes_model.pkl', 'rb') as file:
    nb_classifier = pickle.load(file)

# Load TfidfVectorizer
with open('tfidf_vectorizer.pkl', 'rb') as file:
    tfidf_vectorizer = pickle.load(file)

# Function to predict whether the news is fake or real
def predict_news(text):
    text_matrix = tfidf_vectorizer.transform([text])
    prediction = nb_classifier.predict(text_matrix)
    if prediction[0] == 0:
        return 'REAL'
    elif prediction[0] == 1:
        return 'FAKE'

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/result', methods=['POST'])
def result():
    if request.method == 'POST':
        text_input = request.form['text_input']
        prediction = predict_news(text_input)
        return render_template('result.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)

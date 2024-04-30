import streamlit as st
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

# Load model Naive Bayes dan TfidfVectorizer
with open('naive_bayes_model.pkl', 'rb') as file:
    nb_classifier = pickle.load(file)

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

# Streamlit app
def main():
    st.title('Hoax Detection')
    st.write('Masukkan teks berita untuk memeriksa apakah berita tersebut hoaks atau tidak.')
    
    # Input teks berita
    text_input = st.text_area('Masukkan Teks Berita:')
    
    # Tombol untuk memeriksa
    if st.button('Deteksi'):
        if text_input:
            prediction = predict_news(text_input)
            st.write('Prediksi:', prediction)
        else:
            st.write('Masukkan teks berita terlebih dahulu.')

if __name__ == '__main__':
    main()

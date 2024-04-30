import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle

# Load dataset
df = pd.read_csv('hoax_fix2.csv')

# Mengganti nilai NaN dengan string kosong
df['text_fix'].fillna('', inplace=True)

# Split dataset into features and target
X = df['text_fix']
y = df['label']

# Initialize TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)

# Transform text data into tf-idf features
X = tfidf_vectorizer.fit_transform(X)

# Initialize and train Naive Bayes model
nb_classifier = MultinomialNB()
nb_classifier.fit(X, y)

# Save model and TfidfVectorizer
with open('naive_bayes_model.pkl', 'wb') as file:
    pickle.dump(nb_classifier, file)

with open('tfidf_vectorizer.pkl', 'wb') as file:
    pickle.dump(tfidf_vectorizer, file)

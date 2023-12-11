import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import re
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from itertools import chain
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import streamlit as st

# Function for text cleaning
def cleaning(text):
    # HTML Tag Removal
    text = re.compile('<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});').sub('', str(text))
    # Case folding
    text = text.lower()
    # Trim text
    text = text.strip()
    # Remove punctuations, karakter spesial, and spasi ganda
    text = re.compile('<.*?>').sub('', text)
    text = re.compile('[%s]' % re.escape(string.punctuation)).sub(' ', text)
    text = re.sub('\s+', ' ', text)
    # Number removal
    text = re.sub(r'\[[0-9]*\]', ' ', text)
    text = re.sub(r'[^\w\s]', '', str(text).lower().strip())
    text = re.sub(r'\d', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    # Mengubah text 'nan' dengan whitespace agar nantinya dapat dihapus
    text = re.sub('nan', '', text)
    return text

# Load the preprocessed data
data = pd.read_csv('https://raw.githubusercontent.com/ranianuraini/PencarianPenambanganWeb/main/DataOlah_Antara.csv')

# Drop any rows with missing values
data.dropna(inplace=True)

# Separate features (X) and labels (y)
X = data['artikel_tokens']
y = data['Label']

# Tokenizing and cleaning the input data
data['Judul'] = data['Judul'].apply(lambda x: cleaning(x))
data['Artikel'] = data['Artikel'].apply(lambda x: cleaning(x))
data['judul_tokens'] = data['Judul'].apply(lambda x: word_tokenize(x))
data['artikel_tokens'] = data['Artikel'].apply(lambda x: word_tokenize(x))
stop_words = set(chain(stopwords.words('indonesian'), stopwords.words('english')))
data['judul_tokens'] = data['judul_tokens'].apply(lambda x: [w for w in x if not w in stop_words])
data['artikel_tokens'] = data['artikel_tokens'].apply(lambda x: [w for w in x if not w in stop_words])

# Vectorizing
vectorizer = TfidfVectorizer()
X_tfidf = vectorizer.fit_transform(X)

# Training Naive Bayes classifier
nb_classifier = MultinomialNB()
nb_classifier.fit(X_tfidf, y)

# Streamlit app
def predict_category(input_text):
    input_text = cleaning(input_text)
    input_tokens = word_tokenize(input_text)
    input_tokens = [w for w in input_tokens if not w in stop_words]
    input_tfidf = vectorizer.transform([' '.join(input_tokens)])
    prediction = nb_classifier.predict(input_tfidf)[0]
    return prediction

def main():
    st.title('News Category Prediction App')
    st.write("This app predicts the category of news articles based on the provided data.")

    user_input = st.text_area("Enter a news article:", "")
    if st.button("Predict"):
        if user_input:
            prediction = predict_category(user_input)
            st.success(f"The predicted category is: {prediction}")
        else:
            st.warning("Please enter a news article.")

if __name__ == '__main__':
    main()

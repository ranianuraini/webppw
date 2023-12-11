# Import library
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn import metrics

# Load data
url = "https://raw.githubusercontent.com/ranianuraini/PencarianPenambanganWeb/main/DataOlah_Antara.csv"
df = pd.read_csv(url)

# Preprocess data
X = df['judul']
y = df['kategori']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build Naive Bayes model
model = make_pipeline(CountVectorizer(), MultinomialNB())
model.fit(X_train, y_train)

# Streamlit app
def predict_category(title):
    prediction = model.predict([title])
    return prediction[0]

# Streamlit UI
st.title("Aplikasi Prediksi Kategori Berita")

# Input judul berita
title_input = st.text_input("Masukkan judul berita:")

# Button untuk memprediksi
if st.button("Prediksi"):
    if title_input:
        prediction = predict_category(title_input)
        st.success(f"Kategori berita yang diprediksi: {prediction}")
    else:
        st.warning("Silakan masukkan judul berita terlebih dahulu.")

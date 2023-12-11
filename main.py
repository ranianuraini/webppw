import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

# Load data
data_url = "https://raw.githubusercontent.com/ranianuraini/PencarianPenambanganWeb/main/DataOlah_Antara.csv"
df = pd.read_csv(data_url)

# Preprocessing
X = df['isi_berita']
y = df['kategori']

# Build a classifier
model = make_pipeline(TfidfVectorizer(), MultinomialNB())
model.fit(X, y)

# Streamlit App
def predict_category(news_text):
    prediction = model.predict([news_text])
    return prediction[0]

st.title("Aplikasi Prediksi Kategori Berita")

news_text = st.text_area("Masukkan teks berita di sini:")

if st.button("Prediksi"):
    if news_text:
        prediction = predict_category(news_text)
        st.success(f"Kategori berita diprediksi sebagai: {prediction}")
    else:
        st.warning("Silakan masukkan teks berita terlebih dahulu.")

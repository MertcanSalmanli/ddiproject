
import pandas as pd
import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
from nltk.tokenize import word_tokenize
from nltk.corpus import stop_words
from gensim.models import Word2Vec
import re
from stanza.nlp import Pipeline

# önişleme
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    lemmatized_tokens = []
    for token in tokens:
        doc = nlp(token)
        for sent in doc.sentences:
            for word in sent.words:
                lemmatized_tokens.append(word.lemma)
    return lemmatized_tokens

# vektörize etme
model = Word2Vec(sentences=haber_metni, vector_size=100, window=5, min_count=1, workers=4)
def vectorize_text(text, model):
    words = text.split()
    word_vectors = [model.wv[word] for word in words if word in model.wv]
    if len(word_vectors) > 0:
        return np.mean(word_vectors, axis=0)
    else:
        return np.zeros(model.vector_size)


# Model ve label encoder'ı yükleme
model = load_model('mlp_model.h5')
label_encoder = joblib.load('label_encoder.pkl')

# Streamlit uygulaması
st.title("Haber Kategorisi Tahmini")

# Metin girdisi için alan oluşturma
haber_metni = st.text_input("Haber Metnini Giriniz:")

# Tahmin butonu
if st.button("Tahmin Yap"):
    # Metni ön işleme
    # ... (Metin ön işleme işlemlerini ekleyin)
    
    
    # Metni sayısal diziye dönüştürme
    # ... (Metni sayısal diziye dönüştürme işlemlerini ekleyin)
    X_test=vectorize_text(preprocess_text(haber_metni))
    
    # Tahmini yapma
    tahmin = model.predict(X_test)
    tahmin = np.argmax(tahmin)

    # Tahmini kategoriyi deşifre etme
    tahmin_kategorisi = label_encoder.inverse_transform([tahmin])[0]

    # Sonucu ekrana yazdırma
    st.write(f"Tahmin Edilen Kategori: {tahmin_kategorisi}")

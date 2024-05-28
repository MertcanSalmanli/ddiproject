import pandas as pd
import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re
import stanza
import joblib

# NLTK Türkçe stopwords'leri indir

# Türkçe stopwords'ler

# Stanza pipeline'ını indir
stanza.download('tr')
nlp = stanza.Pipeline(lang='tr', processors='tokenize,mwt,pos,lemma')

# Metin önişleme fonksiyonu
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
    return ' '.join(lemmatized_tokens)

# Vektörize etme fonksiyonu
def vectorize_text(text, model):
    words = text.split()
    word_vectors = [model.wv[word] for word in words if word in model.wv]
    if len(word_vectors) > 0:
        return np.mean(word_vectors, axis=0)
    else:
        return np.zeros(model.vector_size)

# Model ve label encoder'ı yükleme
mlp_model = load_model('mlp_model.h5')
label_encoder = joblib.load('label_encoder.pkl')

# Gensim Word2Vec modelini yükleme
word2vec_model = Word2Vec.load('word2vec_model.bin')

# Streamlit uygulaması
st.title("Haber Kategorisi Tahmini")

# Metin girdisi için alan oluşturma
haber_metni = st.text_input("Haber Metnini Giriniz:")

# Tahmin butonu
if st.button("Tahmin Yap"):
    # Metni ön işleme
    islemli_metin = preprocess_text(haber_metni)
    
    # Metni sayısal diziye dönüştürme
    X_test = vectorize_text(islemli_metin, word2vec_model)
    X_test = X_test.reshape(1, -1)  # Model için uygun şekle getirme
    
    # Tahmini yapma
    tahmin = mlp_model.predict(X_test)
    tahmin = np.argmax(tahmin)

    # Tahmini kategoriyi deşifre etme
    tahmin_kategorisi = label_encoder.inverse_transform([tahmin])[0]

    # Sonucu ekrana yazdırma
    st.write(f"Tahmin Edilen Kategori: {tahmin_kategorisi}")

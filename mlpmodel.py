import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import load_model
import joblib
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import stanza
from gensim.models import Word2Vec

# NLTK Türkçe stopwords ve tokenizerlar
nltk.download('punkt')
nltk.download('stopwords')

# Türkçe stopwords'ler
stop_words = set(stopwords.words('turkish'))

# Stanza 
stanza.download('tr')
nlp = stanza.Pipeline(lang='tr', processors='tokenize,mwt,pos,lemma')

# Haber metnini işlemek için bir fonksiyon
def haber_metnini_islemle(haber_metni):
    # Gerekli ön işlemleri yapın ve vektörleştirin
    text = haber_metni.lower()
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
    # Burada vektörleştirme işlemini yapın
    word_vectors = [model.wv[word] for word in lemmatized_tokens if word in model.wv]
    if len(word_vectors) > 0:
        return np.mean(word_vectors, axis=0)
    else:
        return np.zeros(model.vector_size)

# Streamlit uygulamasını oluşturma
st.title("Haber Türü Tahmini")

# Haber metnini giriş alanı
haber_metni = st.text_area("Haber Metni", "")

# Tahmin et butonu
if st.button("Tahmin Et"):
    # Tahmin etme işlemi
    tahmin = haber_turunu_tahmin_et(haber_metni)
    
    # Tahmini sonucu kullanıcıya gösterme
    st.write("Tahmin Edilen Haber Türü:", tahmin)

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import load_model
import joblib
import streamlit as st

# Modeli ve label encoder'ı yükleme
mlp_model = load_model('mlp_model.h5')
label_encoder = joblib.load('label_encoder.pkl')

# Fonksiyonlar



def haber_turunu_tahmin_et(haber_metni):
  # Metni işleme
  islemli_metin = haber_metnini_islemle(haber_metni)
  
  # Kelimeleri sayısal değerlere dönüştürme
  sayisal_metin = label_encoder.transform(islemli_metin)
  
  # Tahmini yapma
  tahmin = mlp_model.predict_classes(np.array([sayisal_metin]))[0]
  tahmin = label_encoder.inverse_transform([tahmin])[0]
  return tahmin

# Streamlit uygulaması

st.title("Haber Türü Tahmini")

haber_metni = st.text_input("Haber metnini giriniz:")

if st.button("Tahmin Et"):
  tahmin = haber_turunu_tahmin_et(haber_metni)
  st.write("Tahmin edilen haber türü:", tahmin)

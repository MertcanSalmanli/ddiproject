import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
import streamlit as st

# Modelin yüklenmesi
model = Sequential()
# ... (Modelin mlpmodel.ipynb dosyasında tanımlanan katmanları) ...
model.load_weights('mlpmodel.h5')

# Label Encoder'ın yüklenmesi
label_encoder = LabelEncoder()
label_encoder.load_pickle('haber_kategorileri.pickle')

# Fonksiyonlar

def haber_metnini_islemle(haber_metni):
  # Metni ön işleme
  metin = haber_metni.lower()  # Metni küçük harf yapma
  metin = metin.replace('[URL]', '')  # URL'leri kaldırma
  metin = metin.replace('[IMG]', '')  # Görselleri kaldırma
  metin = metin.replace('...', '')  # Elipsleri kaldırma
  metin = metin.replace('"', '')  # Tırnak işaretlerini kaldırma
  metin = metin.replace("'", '')  # Tek tırnak işaretlerini kaldırma
  metin = metin.replace(':', '')  # İki nokta işaretlerini kaldırma
  metin = metin.replace(';', '')  # Virgül işaretlerini kaldırma
  metin = metin.replace('!', '')  # Ünlem işaretlerini kaldırma
  metin = metin.replace('?', '')  # Soru işaretlerini kaldırma
  metin = metin.replace('(', '')  # Parantezleri kaldırma
  metin = metin.replace(')', '')  # Parantezleri kaldırma
  metin = metin.replace('{', '')  # Küme parantezlerini kaldırma
  metin = metin.replace('}', '')  # Küme parantezlerini kaldırma
  metin = metin.replace('[', '')  *Köşeli parantezleri kaldırma*
  metin = metin.replace(']', '')  *Köşeli parantezleri kaldırma*
  metin = metin.replace('<', '')  *Küçük ayraçları kaldırma*
  metin = metin.replace('>', '')  *Büyük ayraçları kaldırma*
  metin = metin.replace('|', '')  *Çubuk işaretlerini kaldırma*
  metin = metin.replace('\\', '')  *Ters eğik çizgileri kaldırma*
  metin = metin.replace('/', '')  *Eğik çizgileri kaldırma*
  metin = metin.replace('—', '')  *Uzun çizgiyi kaldırma*
  metin = metin.replace('–', '')  *Uzun çizgiyi kaldırma*
  metin = metin.replace('…', '')  *Elipsleri kaldırma*
  metin = metin.split()  # Kelimeleri listeye ayırma
  return metin

def haber_turunu_tahmin_et(haber_metni):
  # Metni işleme
  islemli_metin = haber_metnini_islemle(haber_metni)
  
  # Kelimeleri sayısal değerlere dönüştürme
  sayisal_metin = label_encoder.transform(islemli_metin)
  
  # Tahmini yapma
  tahmin = model.predict_classes(np.array([sayisal_metin]))[0]
  tahmin = label_encoder.inverse_transform([tahmin])[0]
  return tahmin

# Streamlit uygulaması

st.title("Haber Türü Tahmini")

haber_metni = st.text_input("Haber metnini giriniz:")

if st.button("Tahmin Et"):
  tahmin = haber_turunu_tahmin_et(haber_metni)
  st.write("Tahmin edilen haber türü:", tahmin)

import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import joblib

# Modeli ve label encoder'ı yükleme
mlp_model = load_model('mlp_model.h5')
label_encoder = joblib.load('label_encoder.pkl')

# Streamlit arayüzü
st.title('Haber Etiketleme Tahmini')
input_text = st.text_area("Bir haber yazısı girin ve hangi etiketten olduğunu tahmin edin:")

if st.button('Tahmin Et'):
    if input_text:
        # Giriş metnini vektöre dönüştürme
        input_vector = np.array([input_text.split()])  # Basitçe metni kelimelere ayırıyoruz
        
        # Dönüştürülmüş girdiyi modelin beklediği forma getirme
        input_vector = input_vector.reshape((input_vector.shape[0], -1))
        
        # Tahmin yapma
        prediction = mlp_model.predict(input_vector)
        predicted_label = np.argmax(prediction)
        
        # Sayısal etiketi gerçek etikete dönüştürme
        predicted_class = label_encoder.inverse_transform([predicted_label])[0]
        
        # Tahmin sonucunu gösterme
        st.write(f"Tahmin edilen etiket: {predicted_class}")
    else:
        st.write("Lütfen bir haber yazısı girin.")

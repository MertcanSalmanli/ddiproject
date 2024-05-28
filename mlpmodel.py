import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
from tensorflow.keras.utils import to_categorical
from sklearn.feature_extraction.text import CountVectorizer

# CSV dosyasından veriyi yükleme
data = pd.read_csv('vektorlenen_dosya.csv')

# Etiketler ve özelliklerin ayrılması
etiketler = data['Etiket']
X_texts = data['Text']  # Metin verisini alıyoruz

# Etiketleri sayısal değerlere dönüştürme
label_encoder = LabelEncoder()
etiketler = label_encoder.fit_transform(etiketler)
num_classes = len(np.unique(etiketler))
etiketler = to_categorical(etiketler)

# Metin verilerini vektörleştirme
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(X_texts).toarray()

# Verileri eğitim ve test setlerine ayırma
X_train, X_test, y_train, y_test = train_test_split(X, etiketler, test_size=0.2, random_state=42)

# LSTM modeline uygun forma getirme
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# LSTM modeli
model = Sequential()
model.add(Input(shape=(X_train.shape[1], 1)))  # Giriş katmanı
model.add(LSTM(100))
model.add(Dense(num_classes, activation='softmax'))

# Modeli uygula
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Modeli eğitme
model.fit(X_train, y_train, epochs=30, batch_size=32, validation_split=0.2)

# Modeli değerlendirme
loss, accuracy = model.evaluate(X_test, y_test)
st.write("Test Loss:", loss)
st.write("Test Accuracy:", accuracy)

# Streamlit arayüzü
st.title('Haber Etiketleme Tahmini')
input_text = st.text_area("Bir haber yazısı girin ve hangi etikete ait olduğunu tahmin edin:")

if st.button('Tahmin Et'):
    if input_text:
        # Yazıyı vektörize etme
        input_vector = vectorizer.transform([input_text]).toarray()
        
        # Modelin girişine uygun hale getirme
        input_vector = input_vector.reshape((input_vector.shape[0], input_vector.shape[1], 1))
        
        # Tahmin yapma
        prediction = model.predict(input_vector)
        predicted_label = np.argmax(prediction)
        predicted_class = label_encoder.inverse_transform([predicted_label])
        
        st.write(f"Tahmin edilen etiket: {predicted_class[0]}")
    else:
        st.write("Lütfen bir haber yazısı girin.")

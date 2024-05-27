#--LSTM MODEL EĞİTİMİ--


import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

# Vektörleştirilmiş verilere sahip CSV dosyasını yükleme
vector_df = pd.read_csv('vektorlenen_dosya.csv')

# Etiketler ve vektörleri birleştirme
etiketler = vector_df['Etiket']
X = vector_df.drop('Etiket', axis=1).values

# Etiketleri sayısal değerlere dönüştürme
label_encoder = LabelEncoder()
etiketler = label_encoder.fit_transform(etiketler)

# İkili sınıflandırma mı çok sınıflı sınıflandırma mı olduğunu kontrol etme (değişti)
num_classes = len(np.unique(etiketler))
etiketler = to_categorical(etiketler)
loss_function = 'categorical_crossentropy'
activation_function = 'softmax'
output_units = num_classes


# Verileri uygun forma dönüştürme
X = X.astype(float)  
X = X.reshape((X.shape[0], X.shape[1], 1))

# eğitim ve test setleri
X_train, X_test, y_train, y_test = train_test_split(X, etiketler, test_size=0.2, random_state=42)

# LSTM modeli
model = Sequential()
model.add(LSTM(100, input_shape=(X_train.shape[1], 1)))
model.add(Dense(output_units, activation=activation_function))

# Modeli uygula
model.compile(optimizer='adam', loss=loss_function, metrics=['accuracy'])

# Modeli eğitme
model.fit(X_train, y_train, epochs=30, batch_size=32, validation_split=0.2)

# Modeli değerlendirme
loss, accuracy = model.evaluate(X_test, y_test)
print("Test Loss:", loss)
print("Test Accuracy:", accuracy)
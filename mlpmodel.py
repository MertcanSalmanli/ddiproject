#--MLP MODEL EĞİTİMİ--


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten

# CSV dosyasından veriyi yükleme
data = pd.read_csv('vektorlenen_dosya.csv')

# Etiketler ve özelliklerin ayrılması
etiketler = data['Etiket']
X = data.drop('Etiket', axis=1).values

# Etiketleri sayısal değerlere dönüştürme
label_encoder = LabelEncoder()
etiketler = label_encoder.fit_transform(etiketler)

# Verileri eğitim ve test setlerine ayırma
X_train, X_test, y_train, y_test = train_test_split(X, etiketler, test_size=0.2, random_state=42)

# MLP modeli
mlp_model = Sequential()
mlp_model.add(Flatten(input_shape=(X_train.shape[1],)))  # Giriş 
mlp_model.add(Dense(128, activation='relu'))  # Gizli 
mlp_model.add(Dense(np.max(etiketler) + 1, activation='softmax'))  # Çıkış 

# Modeli uygula
mlp_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Modeli eğitme
mlp_model.fit(X_train, y_train, epochs=30, batch_size=32, validation_split=0.2)

# Modeli değerlendirme
mlp_loss, mlp_accuracy = mlp_model.evaluate(X_test, y_test)
print("Test Loss (MLP):", mlp_loss)
print("Test Accuracy (MLP):", mlp_accuracy)

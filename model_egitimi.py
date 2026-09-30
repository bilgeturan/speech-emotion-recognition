import pandas as pd
import numpy as np
import os
import pickle
from tensorflow.keras.callbacks import CSVLogger

# Kütüphaneler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Dropout

# 1. VERİYİ YÜKLE
print("Veriler yukleniyor...")
df = pd.read_csv("islenmis_veriler.csv")

X = df.iloc[:, 1:].values 
y = df['label'].values    

# 2. VERİYİ HAZIRLA
encoder = LabelEncoder()
y = encoder.fit_transform(y)

with open("etiket_sozlugu.pkl", "wb") as f:
    pickle.dump(encoder, f)

y = to_categorical(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Scaler'ı (Cetveli) kaydediyoruz ki app.py'de kullanabilelim!
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
print("✅ Scaler (Ölçekleyici) kaydedildi!")
# ----------------------------------

X_train = np.expand_dims(X_train, axis=2)
X_test = np.expand_dims(X_test, axis=2)

# 3. MODELİ KUR
model = Sequential()
model.add(Conv1D(64, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1)))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.2))

model.add(Conv1D(128, kernel_size=3, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(y.shape[1], activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
input("📸 Tabloyu çekmek için durdum! Devam etmek için Enter'a bas...")
print("Model egitimi basliyor!")
# Log dosyasını tanımla
csv_logger = CSVLogger('egitim_loglari.csv', append=False)

# Eğitimi başlat (callbacks kısmına dikkat)
history = model.fit(X_train, y_train, batch_size=32, epochs=50, validation_data=(X_test, y_test), callbacks=[csv_logger])
print("Egitim bitti! Model kaydediliyor...")
model.save("duygu_tanima_modeli.h5")
print(f"Model basarisi: %{history.history['accuracy'][-1]*100:.2f}")
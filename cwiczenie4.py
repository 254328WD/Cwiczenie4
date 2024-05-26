import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input

# Wczytywanie danych z pliku CSV
data_path = "D:\\studia wojtek\\2 stopien\\semestr 1\\Programowanie w obliczeniach inteligentnych\\cwiczenie4\\cwiczenie4\\texture_features.csv"
data = pd.read_csv(data_path)

# Sprawdzenie struktury pliku
print(data.head())

# Wyodrębnienie wektorów cech do macierzy X oraz etykiet kategorii do wektora y
# Pominięcie pierwszej kolumny z nazwami plików
X = data.iloc[:, 2:].values  # Pominięcie pierwszych dwóch kolumn
y = data.iloc[:, 1].values  # Druga kolumna jako etykiety

# Wstępne przetwarzanie danych
# Kodowanie całkowitoliczbowe dla wektora y
label_encoder = LabelEncoder()
y_int = label_encoder.fit_transform(y)

# Kodowanie 1 z n dla wektora y_int
onehot_encoder = OneHotEncoder(sparse_output=False)
y_onehot = onehot_encoder.fit_transform(y_int.reshape(-1, 1))

# Konwersja danych do typu float32
X = X.astype('float32')
y_onehot = y_onehot.astype('float32')

# Podzielenie zbioru na część treningową (70%) i testową (30%)
X_train, X_test, y_train, y_test = train_test_split(X, y_onehot, test_size=0.3, random_state=42)

# Tworzenie modelu sieci neuronowej
model = Sequential()
model.add(Input(shape=(X_train.shape[1],)))
model.add(Dense(10, activation='sigmoid'))
model.add(Dense(3, activation='softmax'))

# Kompilacja modelu
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

# Uczenie sieci
model.fit(X_train, y_train, epochs=100, batch_size=10, verbose=1)

# Testowanie sieci
y_pred = model.predict(X_test)
y_test_int = label_encoder.inverse_transform(y_test.argmax(axis=1))
y_pred_int = label_encoder.inverse_transform(y_pred.argmax(axis=1))

# Macierz pomyłek
conf_matrix = confusion_matrix(y_test_int, y_pred_int)
print("Macierz pomyłek:\n", conf_matrix)

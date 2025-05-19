import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, GlobalMaxPool1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import joblib

df = pd.read_csv("reformatted_movies.csv")
X = df['description'].fillna("")
y = df['genres'].apply(lambda x: [g.strip() for g in x.split(',')])

mlb = MultiLabelBinarizer()
Y = mlb.fit_transform(y)
joblib.dump(mlb, "models/label_binarizer.joblib")

tokenizer = Tokenizer(num_words=10000, oov_token='<OOV>')
tokenizer.fit_on_texts(X)
X_seq = tokenizer.texts_to_sequences(X)
X_pad = pad_sequences(X_seq, maxlen=200)

X_train, X_test, y_train, y_test = train_test_split(X_pad, Y, test_size=0.2, random_state=42)

model = Sequential([
    Embedding(10000, 128, input_length=200),
    LSTM(64, return_sequences=True),
    GlobalMaxPool1D(),
    Dense(64, activation='relu'),
    Dense(Y.shape[1], activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=5, batch_size=32, validation_split=0.2)

model.save("models/L1_LSTM.h5")
joblib.dump(tokenizer, "models/lstm_tokenizer.joblib")

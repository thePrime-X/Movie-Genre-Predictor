import streamlit as st
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer

# Define the model names excluding 'L5_BERT'
MODEL_NAMES = ["L1_LSTM", "L2_NaiveBayes", "L3_SVM", "L4_LogisticRegression"]

@st.cache_resource
def load_all():
    # Load the models except for L1_LSTM and L5_BERT (since BERT is removed)
    models = {
        name: joblib.load(f"models/{name}.joblib")
        for name in MODEL_NAMES if name != "L1_LSTM"
    }
    mlb = joblib.load("models/label_binarizer.joblib")
    tfidf = joblib.load("models/bert_encoder.joblib")
    return models, mlb, tfidf

models, mlb, tfidf = load_all()

st.title("🎬 Multi-label Genre Predictor")
desc = st.text_area("Movie/Show Description", height=150)
mode = st.radio("Prediction Mode:", ["Single Model", "Compare All Models"])
selected_model = st.selectbox("Select model:", MODEL_NAMES) if mode == "Single Model" else None

def predict_with_model(model_name, desc):
    if model_name in models:
        # Ensure the vectorizer is applied correctly without 'encode' method
        X_vec = tfidf.transform([desc])  # Corrected: using transform instead of encode
        probs = models[model_name].predict_proba(X_vec)[0]
        return probs
    elif model_name == "L1_LSTM":
        lstm_model = load_model("models/L1_LSTM.h5")
        tokenizer = joblib.load("models/lstm_tokenizer.joblib")
        X_seq = tokenizer.texts_to_sequences([desc])
        X_pad = pad_sequences(X_seq, maxlen=200)
        return lstm_model.predict(X_pad)[0]

if st.button("Predict Genres"):
    if not desc.strip():
        st.warning("Please enter a description.")
    else:
        target_models = [selected_model] if selected_model else MODEL_NAMES
        for model_name in target_models:
            try:
                # Predicting genres using the selected model
                probs = predict_with_model(model_name, desc)
                # Sorting predictions by highest probability
                top_indices = np.argsort(probs)[::-1][:5]
                top_labels = mlb.classes_[top_indices]
                top_probs = probs[top_indices]
                st.subheader(f"🔹 {model_name}")
                for genre, prob in zip(top_labels, top_probs):
                    st.write(f"• {genre} ({prob * 100:.2f}%)")
            except Exception as e:
                st.error(f"Error with {model_name}: {e}")

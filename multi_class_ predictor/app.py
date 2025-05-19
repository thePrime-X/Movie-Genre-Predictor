import streamlit as st
import joblib
import numpy as np
import pandas as pd
import os
from datetime import datetime

MODEL_NAMES = [
    "C1_MultinomialNB", "C2_GaussianNB", "C3_LogisticRegression",
    "C4_DecisionTree", "C5_RandomForest", "C6_XGBoost",
    "C7_LightGBM", "C8_CatBoost", "C9_SVM"
]

LOG_FILE = "prediction_logs.csv"

@st.cache_resource
def load_vectorizer_and_encoder():
    vectorizer = joblib.load("models/tfidf_vectorizer.joblib")
    label_encoder = joblib.load("models/label_encoder.joblib")
    return vectorizer, label_encoder

@st.cache_resource
def load_model(model_name):
    return joblib.load(f"models/{model_name}.joblib")

tfidf, le = load_vectorizer_and_encoder()

st.title("🎬 Netflix Genre Predictor")
st.markdown("Enter a **description**, see **Top-3 genres**, compare across models, and log your prediction.")

user_desc = st.text_area("Movie/Show Description", height=150)

view_mode = st.radio("Choose Prediction Mode:", ["Single Model", "Compare All Models"])

if view_mode == "Single Model":
    model_choice = st.selectbox("Choose a model:", MODEL_NAMES)

if st.button("Predict Genre"):
    if not user_desc.strip():
        st.warning("Please enter a description.")
    else:
        X_input = tfidf.transform([user_desc])
        results = []

        model_list = [model_choice] if view_mode == "Single Model" else MODEL_NAMES

        with st.spinner("Predicting..."):
            for model_name in model_list:
                model = load_model(model_name)

                if model_name == "C2_GaussianNB":
                    y_prob = model.predict_proba(X_input.toarray())[0]
                else:
                    y_prob = model.predict_proba(X_input)[0]

                # Get top 3 genres without duplication
                top_3_indices = np.argsort(y_prob)[::-1]
                top_3_genres = le.inverse_transform(top_3_indices)

                # Ensure no duplicates in top 3 genres
                unique_top_3_genres = []
                unique_confidences = []
                for i in range(len(top_3_genres)):
                    if top_3_genres[i] not in unique_top_3_genres:
                        unique_top_3_genres.append(top_3_genres[i])
                        unique_confidences.append(y_prob[top_3_indices[i]])
                    if len(unique_top_3_genres) == 3:
                        break

                # Make sure we have exactly 3 entries
                while len(unique_top_3_genres) < 3:
                    unique_top_3_genres.append("N/A")
                    unique_confidences.append(0.0)

                results.append({
                    "Model": model_name,
                    "Top Genre": unique_top_3_genres[0],
                    "Top Confidence (%)": round(unique_confidences[0] * 100, 2),
                    "2nd Genre": unique_top_3_genres[1],
                    "2nd Confidence (%)": round(unique_confidences[1] * 100, 2),
                    "3rd Genre": unique_top_3_genres[2],
                    "3rd Confidence (%)": round(unique_confidences[2] * 100, 2)
                })

        df_results = pd.DataFrame(results)
        st.subheader("🎯 Prediction Results")
        st.dataframe(df_results)

        # Save prediction to log file
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "description": user_desc,
        }

        for row in results:
            log_entry[f"{row['Model']}_TopGenre"] = row["Top Genre"]
            log_entry[f"{row['Model']}_Confidence"] = row["Top Confidence (%)"]

        log_df = pd.DataFrame([log_entry])

        if not os.path.exists(LOG_FILE):
            log_df.to_csv(LOG_FILE, index=False)
        else:
            log_df.to_csv(LOG_FILE, mode='a', header=False, index=False)

        st.success("✅ Prediction logged successfully.")

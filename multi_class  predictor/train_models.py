import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import joblib

# Ensure models directory exists
os.makedirs("models", exist_ok=True)

# Load dataset
df = pd.read_csv("reformatted_movies.csv")

# Drop missing genres or description
df = df.dropna(subset=["genres", "description"])

# Encode target (first genre only)
df["genre_main"] = df["genres"].apply(lambda x: x.split(",")[0])
le = LabelEncoder()
y = le.fit_transform(df["genre_main"])

# Save label encoder
joblib.dump(le, "models/label_encoder.joblib")

# Features: we'll use description text
X_raw = df["description"]

# Vectorize description
tfidf = TfidfVectorizer(max_features=5000)
X = tfidf.fit_transform(X_raw)

# Save vectorizer
joblib.dump(tfidf, "models/tfidf_vectorizer.joblib")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define models
models = {
    "C1_MultinomialNB": MultinomialNB(),
    "C2_GaussianNB": GaussianNB(),
    "C3_LogisticRegression": LogisticRegression(max_iter=500),
    "C4_DecisionTree": DecisionTreeClassifier(),
    "C5_RandomForest": RandomForestClassifier(),
    "C6_XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='mlogloss'),
    "C7_LightGBM": LGBMClassifier(),
    "C8_CatBoost": CatBoostClassifier(verbose=10, iterations=600),
    "C9_SVM": SVC(probability=True)
}

# Train and save models
for name, model in models.items():
    if name == "C2_GaussianNB":
        model.fit(X_train.toarray(), y_train)
    else:
        model.fit(X_train, y_train)
    joblib.dump(model, f"models/{name}.joblib")
    print(f"{name} saved.")

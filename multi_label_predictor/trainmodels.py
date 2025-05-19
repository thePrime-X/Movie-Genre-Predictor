import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier

# Load the data
df = pd.read_csv("reformatted_movies.csv")

# Prepare the features (X) and labels (y)
X = df['description'].fillna("")  # Handle missing descriptions
y = df['genres'].apply(lambda x: [g.strip() for g in x.split(',')])  # Clean genres column

# Apply MultiLabelBinarizer to the labels
mlb = MultiLabelBinarizer()
Y = mlb.fit_transform(y)

# Save the label binarizer (for later use in app.py)
joblib.dump(mlb, "models/label_binarizer.joblib")

# Apply TF-IDF Vectorizer to the descriptions
vectorizer = TfidfVectorizer(max_features=5000)
X_vec = vectorizer.fit_transform(X)  # Fit and transform on the training data

# Save the vectorizer (for later use in app.py)
joblib.dump(vectorizer, "models/bert_encoder.joblib")

# Define models wrapped with OneVsRestClassifier for multi-label classification
models = {
    "L2_NaiveBayes": OneVsRestClassifier(MultinomialNB()),
    "L3_SVM": OneVsRestClassifier(SVC(probability=True)),
    "L4_LogisticRegression": OneVsRestClassifier(LogisticRegression(max_iter=300))
}

# Train and save each model
for name, model in models.items():
    model.fit(X_vec, Y)  # Train the model
    joblib.dump(model, f"models/{name}.joblib")  # Save the model
    print(f"{name} saved.")

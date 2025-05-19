# Movie-Genre-Predictor
Streamlit-based app to predict movie/show genres using both multi-class and multi-label models trained on textual descriptions.

---

## 🧠 Models Implemented

### 🔹 Multi-Label Classification (Multiple genres per movie)
- `L1_LSTM`: Deep learning model using Keras (with tokenized and padded sequences)
- `L2_NaiveBayes`
- `L3_SVM`
- `L4_LogisticRegression`

> These models are trained to assign multiple genres per input and return probabilities for each.

---

### 🔸 Multi-Class Classification (Single genre per movie)
- `C1_MultinomialNB`
- `C2_GaussianNB`
- `C3_LogisticRegression`
- `C4_DecisionTree`
- `C5_RandomForest`
- `C6_XGBoost`
- `C7_LightGBM`
- `C8_CatBoost`
- `C9_SVM`

> These models predict a single most likely genre and provide the Top-3 predictions with confidence scores.

---

## 🚀 Features

- ✅ Predict genres from a natural-language movie/show description
- ✅ Choose between single model or compare all models
- ✅ Multi-label (e.g., Action, Sci-Fi, Thriller) and multi-class support
- ✅ View top predicted genres with probability scores
- ✅ Automatically logs predictions with timestamps
- ✅ Built with an intuitive Streamlit interface

---

## 📦 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/thePrime-X/Movie-Genre-Predictor.git
   cd Movie-Genre-Predictor

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
3. **Run the app**
   ```bash
   streamlit run app.py

---

## 📁 Project Structure
```
   Movie-Genre-Predictor/
   ├── multi-label predictor/     # Multi-label predictor directory
   ├── multi-class predictor/     # Multi-class predictor directory
   ├── requirements.txt        # Python dependencies
   └── README.md               # Project documentation
```
---

## ✨ Example Use Case

**Input:**
```text
A former U.S. President is called out of retirement to find the source of a deadly cyberattack, only to    
discover a vast web of lies and conspiracies.
```

**Output (Multi-Label):**
- Sci-Fi (65%)
- Drama (53%)

**Output (Multi-Class):**
- Top Genre: Sci-Fi (89.3%)
- 2nd Genre: Adventure (72.1%)

---















# ==============================
# Import Libraries
# ==============================
import streamlit as st
import pandas as pd
import re
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ==============================
# Page Config
# ==============================
st.set_page_config(
    page_title="Fake Job Posting Detection",
    layout="wide"
)

# ==============================
# Custom CSS (ONLY UI)
# ==============================
st.markdown("""
<style>
/* ---- App background ---- */
[data-testid="stAppViewContainer"] {
    background-color: #1b1b1b;
}

/* ---- Centered container ---- */
.main-container {
    max-width: 1100px;
    margin: auto;
    padding: 30px 20px 40px 20px;
}

/* ---- Header ---- */
.main-title {
    font-size: 50px;
    font-weight: 600;
    text-align: center;
    color: #ffffff;
    margin-bottom: 6px;
}

.subtitle {
    font-size: 14px;
    text-align: center;
    color: #ffffff;
    margin-bottom: 30px;
}

.divider {
    border-top: 5px solid #ffffff;
    margin: 30px 0;
}

/* ---- Section titles ---- */
.section-title {
    font-size: 18px;
    font-weight: 600;
    color: #ffffff;
    margin-bottom: 12px;
}

/* ---- Metrics ---- */
.metric {
    font-size: 24px;
    font-weight: 600;
    color: #2563eb;
}

.metric-label {
    font-size: 13px;
    color: #ffffff;
}

/* ---- Buttons ---- */
.stButton > button {
    background-color: #2563eb;
    color: white;
    border-radius: 6px;
    padding: 8px 16px;
    border: none;
}

.stButton > button:hover {
    background-color: #1e40af;
}

/* ---- Remove empty blocks ---- */
[data-testid="stVerticalBlock"] > div:empty {
    display: none;
}
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='main-container'>", unsafe_allow_html=True)

# ==============================
# Header
# ==============================
st.markdown("<div class='main-title'>Fake Job Posting Detection</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='subtitle'>Detect whether a job posting is real or fake using a Random Forest classifier</div>",
    unsafe_allow_html=True
)

# ==============================
# Paths
# ==============================
MODEL_PATH = "rf_model.joblib"
VECTORIZER_PATH = "tfidf_vectorizer.joblib"
FEEDBACK_FILE = "feedback.csv"
DATASET_PATH = "fake_job_postings.csv"

# ==============================
# Load Dataset
# ==============================
@st.cache_data
def load_dataset(path: str):
    return pd.read_csv(path)

df = load_dataset(DATASET_PATH)

st.markdown("<div class='card'>", unsafe_allow_html=True)
st.markdown("<div class='section-title'>Dataset Preview</div>", unsafe_allow_html=True)
st.dataframe(df.head())
st.markdown("</div>", unsafe_allow_html=True)

# ==============================
# Preprocessing
# ==============================
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(s):
    s = s.lower()
    s = re.sub(r'<[^>]+>', ' ', s)
    s = re.sub(r'http\S+', ' ', s)
    s = re.sub(r'[^a-z\s]', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def preprocess(s):
    tokens = word_tokenize(s)
    tokens = [t for t in tokens if t.isalpha() and t not in stop_words and len(t) > 1]
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return " ".join(tokens)

# ==============================
# Model Training
# ==============================
@st.cache_resource
def train_model(df):
    text_cols = [
        'title','location','department','salary_range','company_profile',
        'description','requirements','benefits','employment_type',
        'required_experience','required_education','industry','function'
    ]

    existing_text_cols = [c for c in text_cols if c in df.columns]
    df[existing_text_cols] = df[existing_text_cols].astype(str).fillna('')
    df['text'] = df[existing_text_cols].agg(' ||| '.join, axis=1)
    df['cleaned'] = df['text'].apply(clean_text)
    df['processed'] = df['cleaned'].apply(preprocess)

    x = df['processed']
    y = df['fraudulent']

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, stratify=y, random_state=42
    )

    tfidf = TfidfVectorizer(max_features=5000)
    x_train_tfidf = tfidf.fit_transform(x_train)
    x_test_tfidf = tfidf.transform(x_test)

    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(x_train_tfidf, y_train)
    y_pred = model.predict(x_test_tfidf)

    results = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred)
    }

    return model, tfidf, results

# ==============================
# Load or Train Model
# ==============================
with st.spinner("Loading/Training model..."):
    if os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
        model = joblib.load(MODEL_PATH)
        tfidf = joblib.load(VECTORIZER_PATH)
        model, tfidf, results = train_model(df)
    else:
        model, tfidf, results = train_model(df)
        joblib.dump(model, MODEL_PATH)
        joblib.dump(tfidf, VECTORIZER_PATH)

# ==============================
# Model Performance
# ==============================
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.markdown("<div class='section-title'>Model Performance</div>", unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)
c1.markdown(f"<div class='metric'>{results['accuracy']*100:.2f}%</div><div class='metric-label'>Accuracy</div>", unsafe_allow_html=True)
c2.markdown(f"<div class='metric'>{results['precision']*100:.2f}%</div><div class='metric-label'>Precision</div>", unsafe_allow_html=True)
c3.markdown(f"<div class='metric'>{results['recall']*100:.2f}%</div><div class='metric-label'>Recall</div>", unsafe_allow_html=True)
c4.markdown(f"<div class='metric'>{results['f1']*100:.2f}%</div><div class='metric-label'>F1 Score</div>", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)

# ==============================
# Prediction
# ==============================
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.markdown("<div class='section-title'>Test with Your Own Job Posting</div>", unsafe_allow_html=True)

user_input = st.text_area("Enter job description / title / requirements", height=150)

if st.button("Predict"):
    if user_input.strip():
        processed = preprocess(clean_text(user_input))
        vec = tfidf.transform([processed])
        pred = model.predict(vec)[0]
        label = "Real Job" if pred == 0 else "Fake Job"
        st.success(f"Prediction: {label}")

        st.session_state["last_input"] = user_input
        st.session_state["last_pred"] = int(pred)
    else:
        st.warning("Please enter job details.")

st.markdown("</div>", unsafe_allow_html=True)

# ==============================
# Feedback System
# ==============================
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.markdown("<div class='section-title'>Feedback</div>", unsafe_allow_html=True)

feedback_choice = st.radio("Was the prediction correct?", ["Correct", "Incorrect"])

if st.button("Save Feedback"):
    if "last_input" in st.session_state:
        df_fb = pd.DataFrame(
            [[st.session_state["last_input"], st.session_state["last_pred"], feedback_choice]],
            columns=["Text", "Prediction", "Feedback"]
        )
        if os.path.exists(FEEDBACK_FILE):
            old = pd.read_csv(FEEDBACK_FILE)
            df_fb = pd.concat([old, df_fb], ignore_index=True)
        df_fb.to_csv(FEEDBACK_FILE, index=False)
        st.info("Feedback saved.")
    else:
        st.warning("Make a prediction first.")

st.markdown("</div>", unsafe_allow_html=True)

# ==============================
# Retraining
# ==============================
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.markdown("<div class='section-title'>Retrain Model</div>", unsafe_allow_html=True)

if st.button("Retrain Model with Feedback"):
    if os.path.exists(FEEDBACK_FILE):
        feedback_df = pd.read_csv(FEEDBACK_FILE)
        incorrect = feedback_df[feedback_df["Feedback"] == "Incorrect"]
        if not incorrect.empty:
            incorrect = incorrect.rename(columns={"Text": "description"})
            incorrect["fraudulent"] = incorrect["Prediction"].apply(lambda x: 1 - x)
            new_df = pd.concat([df, incorrect], ignore_index=True)
            new_df["fraudulent"] = new_df["fraudulent"].astype(int)

            with st.spinner("Retraining model..."):
                model, tfidf, results = train_model(new_df)
                joblib.dump(model, MODEL_PATH)
                joblib.dump(tfidf, VECTORIZER_PATH)
            st.success("Model retrained successfully.")
        else:
            st.info("No incorrect feedback found.")
    else:
        st.warning("Feedback file not found.")

st.markdown("</div>", unsafe_allow_html=True)

# ==============================
# View Feedback
# ==============================
if os.path.exists(FEEDBACK_FILE):
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Recent Feedback</div>", unsafe_allow_html=True)
    st.dataframe(pd.read_csv(FEEDBACK_FILE).tail(10))
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)

# ==============================
# Cache Control (UNCHANGED)
# ==============================
if st.button("Clear Cache & Restart"):
    st.cache_data.clear()
    st.cache_resource.clear()
    st.rerun()

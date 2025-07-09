import streamlit as st
import joblib
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

# Load word lists
with open("positive-words.txt", "r", encoding='ISO-8859-1') as f:
    positive_words = set(line.strip() for line in f if line and not line.startswith(";"))

with open("negative-words.txt", "r", encoding='ISO-8859-1') as f:
    negative_words = set(line.strip() for line in f if line and not line.startswith(";"))

stop_words = set(stopwords.words('english'))

# Text cleaning and feature extraction
def clean_text(text):
    text = re.sub(r'[^\w\s]', '', text.lower())
    tokens = text.split()
    return [word for word in tokens if word not in stop_words]

def extract_features(text):
    tokens = clean_text(text)
    return {
        "positive_count": sum(1 for word in tokens if word in positive_words),
        "negative_count": sum(1 for word in tokens if word in negative_words),
        "total_words": len(tokens)
    }

# Load model and vectorizer
clf = joblib.load("sentiment_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Streamlit UI
st.title("💬 Sentiment Analysis App")
text_input = st.text_area("Enter your text:")

if st.button("Predict Sentiment"):
    feats = extract_features(text_input)
    vect = vectorizer.transform([feats])
    result = clf.predict(vect)[0]
    st.success(f"Predicted Sentiment: **{result.upper()}**")

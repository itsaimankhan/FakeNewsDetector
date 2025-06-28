import streamlit as st
import pandas as pd
import joblib
import pickle
import re

# Load models and vectorizer
@st.cache_resource
def load_resources():
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
    with open("logistic_model.pkl", "rb") as f:
        logistic_model = pickle.load(f)
    with open("naive_bayes_model.pkl", "rb") as f:
        nb_model = pickle.load(f)
    return vectorizer, logistic_model, nb_model

vectorizer, logistic_model, nb_model = load_resources()

# Text preprocessing function
def preprocess_text(text):
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    text = text.lower().strip()
    return text

# Predict function
def predict(text, model_name):
    processed = preprocess_text(text)
    vector = vectorizer.transform([processed])
    if model_name == "Logistic Regression":
        result = logistic_model.predict(vector)[0]
    else:
        result = nb_model.predict(vector)[0]
    return "🟢 Real News" if result == 1 else "🔴 Fake News"

# App title
st.title("📰 Fake News Detector")

# Sidebar options
st.sidebar.title("Choose Prediction Settings")
model_choice = st.sidebar.selectbox("Select Model", ["Logistic Regression", "Naive Bayes"])
input_mode = st.sidebar.radio("Input Mode", ["Enter Text", "Upload File"])

# Text input section
st.header("📝 Input News Article")

user_input = ""
if input_mode == "Enter Text":
    user_input = st.text_area("Paste the news article here:", height=250)
else:
    uploaded_file = st.file_uploader("Upload a .txt file", type=["txt"])
    if uploaded_file is not None:
        user_input = uploaded_file.read().decode("utf-8")
        st.text_area("Extracted Text:", user_input, height=250)

# Prediction section
if st.button("🔍 Predict"):
    if user_input.strip() == "":
        st.warning("Please provide text input before prediction.")
    else:
        prediction = predict(user_input, model_choice)
        st.success(f"**Prediction ({model_choice}): {prediction}**")

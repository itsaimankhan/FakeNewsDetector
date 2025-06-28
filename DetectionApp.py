# Importing necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer

# Setting page title
st.set_page_config(page_title="Fake News Detector", layout="wide")
st.title("📰 Fake News Detector")

# Loading dataset
@st.cache_data
def load_data():
    true_data = pd.read_csv("True.csv", encoding="utf-8")
    fake_data = pd.read_csv("Fake.csv", encoding="utf-8")
    true_data['label'] = 1  # True News
    fake_data['label'] = 0  # Fake News
    return pd.concat([true_data, fake_data]).reset_index(drop=True)

data = load_data()

# Loading models and vectorizer
@st.cache_resource
def load_models():
    vectorizer = joblib.load('tfidf_vectorizer.pkl')
    with open('logistic_model.pkl', 'rb') as f:
        lr_model = pickle.load(f)
    with open('naive_bayes_model.pkl', 'rb') as f:
        nb_model = pickle.load(f)
    return vectorizer, lr_model, nb_model

vectorizer, lr_model, nb_model = load_models()

# Sidebar Navigation
st.sidebar.title("📂 Navigation")
section = st.sidebar.radio("Go to:", ["Introduction", "EDA", "Model", "Try It Yourself", "Conclusion"])

if section == "Introduction":
    st.header("📘 Introduction")
    st.write("The Fake News Detector classifies news articles as fake or real using Logistic Regression and Naive Bayes models.")
    st.write("Dataset: Kaggle - Fake.csv (23,502 articles) & True.csv (21,417 articles)")
    st.write("Columns: `title`, `text`, `subject`, `date`")

elif section == "EDA":
    st.header("📊 Exploratory Data Analysis")
    st.subheader("Sample Data")
    st.dataframe(data.sample(5))

    st.subheader("Class Distribution")
    fig, ax = plt.subplots()
    sns.countplot(x=data['label'], palette='pastel', ax=ax)
    ax.set_xticklabels(["Fake News", "True News"])
    ax.set_ylabel("Count")
    st.pyplot(fig)

    st.subheader("Text Length Distribution")
    data['text_length'] = data['text'].apply(len)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(data['text_length'], bins=30, kde=True, color='purple', ax=ax)
    st.pyplot(fig)

    st.subheader("Word Cloud of Text Data")
    text_data = ' '.join(data['text'].astype(str))
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text_data)
    st.image(wordcloud.to_array(), use_column_width=True)

    st.subheader("Top 10 Most Frequent Words")
    cv = CountVectorizer(stop_words='english', max_features=10)
    cv_fit = cv.fit_transform(data['text'].astype(str))
    top_words = cv.get_feature_names_out()
    word_freq = cv_fit.sum(axis=0).A1
    word_freq_dict = dict(zip(top_words, word_freq))
    top_words_df = pd.DataFrame(sorted(word_freq_dict.items(), key=lambda x: x[1], reverse=True), columns=["Word", "Frequency"])
    st.write(top_words_df)

elif section == "Model":
    st.header("📈 Model Training & Evaluation")
    X = data['text']
    y = data['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    vectorizer_new = TfidfVectorizer(stop_words='english')
    X_train_vec = vectorizer_new.fit_transform(X_train)
    X_test_vec = vectorizer_new.transform(X_test)

    lr_model_new = LogisticRegression()
    lr_model_new.fit(X_train_vec, y_train)
    nb_model_new = MultinomialNB()
    nb_model_new.fit(X_train_vec, y_train)

    st.subheader("Accuracy Scores")
    scores = pd.DataFrame({
        "Model": ["Logistic Regression", "Naive Bayes"],
        "Training Accuracy": [
            accuracy_score(y_train, lr_model_new.predict(X_train_vec)),
            accuracy_score(y_train, nb_model_new.predict(X_train_vec))
        ],
        "Testing Accuracy": [
            accuracy_score(y_test, lr_model_new.predict(X_test_vec)),
            accuracy_score(y_test, nb_model_new.predict(X_test_vec))
        ]
    })
    st.table(scores)

    st.subheader("Confusion Matrices")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.heatmap(confusion_matrix(y_test, lr_model_new.predict(X_test_vec)), annot=True, fmt='d', cmap='Blues', ax=axes[0])
    axes[0].set_title('Logistic Regression')
    sns.heatmap(confusion_matrix(y_test, nb_model_new.predict(X_test_vec)), annot=True, fmt='d', cmap='Greens', ax=axes[1])
    axes[1].set_title('Naive Bayes')
    st.pyplot(fig)

elif section == "Try It Yourself":
    st.header("🧪 Try It Yourself")
    model_choice = st.selectbox("Choose a Model", ["Logistic Regression", "Naive Bayes"])
    input_option = st.radio("Input Type", ["Type/Paste Text", "Upload Text File"])

    def preprocess_text(text):
        text = re.sub(r'[^a-zA-Z]', ' ', text)
        return text.lower().strip()

    article_text = ""
    if input_option == "Type/Paste Text":
        article_text = st.text_area("Enter the news article text:")
    else:
        uploaded_file = st.file_uploader("Upload a .txt file", type=["txt"])
        if uploaded_file:
            article_text = uploaded_file.read().decode("utf-8")

    if st.button("Predict"):
        if not article_text.strip():
            st.warning("Please provide news text input.")
        else:
            processed_text = preprocess_text(article_text)
            vector = vectorizer.transform([processed_text])
            if model_choice == "Logistic Regression":
                pred = lr_model.predict(vector)[0]
            else:
                pred = nb_model.predict(vector)[0]
            label = "✅ True News" if pred == 1 else "❌ Fake News"
            st.subheader("Prediction Result")
            st.success(label)

elif section == "Conclusion":
    st.header("📌 Conclusion")
    st.write("Logistic Regression outperforms Naive Bayes in accuracy and F1-score on this dataset.")
    st.write("Logistic Regression is more precise and stable, while Naive Bayes is faster but may misclassify in some scenarios.")
    st.write("This app offers both insights and real-time predictions to help identify misinformation.")

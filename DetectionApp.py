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

# Setting page title
st.title("Fake News Detector")

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
st.sidebar.title("Navigation")
section = st.sidebar.radio("Go to:", ["Introduction", "EDA", "Model", "Conclusion"])

if section == "Introduction":
    st.header("Introduction")
    st.write("The Fake News Detector project aims to develop a machine learning-based system to classify news articles as either fake or real. This project uses Logistic Regression and Naive Bayes models to classify news articles based on features extracted from the text.")
    st.write("Dataset: The dataset used in this project is from Kaggle and consists of news articles with their respective labels (real or fake). The dataset includes text data that was preprocessed to remove unnecessary noise, and features were extracted for model training.")
    st.write("Dataset separated in two files: ")
    st.write("Fake.csv (23502 fake news articles)")
    st.write("True.csv (21417 true news articles)")
    st.write("Dataset columns: Title: title of the news article, Text: body text of the news article, Subject: subject of the news article, Date: publish date of the news article")

elif section == "EDA":
    st.header("Exploratory Data Analysis (EDA)")
    st.write("### Data Preview")
    st.dataframe(data.sample(5))

    #ClassDistribution
    st.write("### Class Distribution")
    fig, ax = plt.subplots()
    sns.countplot(x=data['label'], palette='pastel', ax=ax)
    ax.set_xticklabels(["Fake News", "True News"])
    ax.set_ylabel("Count")
    st.pyplot(fig)

    #Text Length Distribution
    st.write("### Text Length Distribution")
    data['text_length'] = data['text'].apply(len)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(data['text_length'], bins=30, kde=True, color='purple', ax=ax)
    ax.set_title('Distribution of Text Lengths')
    ax.set_xlabel('Text Length')
    ax.set_ylabel('Frequency')
    st.pyplot(fig)

    #Word Cloud Visualization
    from wordcloud import WordCloud

    st.write("### Word Cloud of Text Data")
    text_data = ' '.join(data['text'].values)
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text_data)
    st.image(wordcloud.to_array(), use_column_width=True)
    st.write("### Top 10 Most Frequent Words")
    from sklearn.feature_extraction.text import CountVectorizer
    cv = CountVectorizer(stop_words='english', max_features=10)
    cv_fit = cv.fit_transform(data['text'])
    top_words = cv.get_feature_names_out()
    word_freq = cv_fit.sum(axis=0).A1
    word_freq_dict = dict(zip(top_words, word_freq))
    sorted_words = sorted(word_freq_dict.items(), key=lambda x: x[1], reverse=True)
    top_words_df = pd.DataFrame(sorted_words, columns=["Word", "Frequency"])
    st.write(top_words_df)

elif section == "Model":
    st.header("Model Predictions and Evaluation Results")

    # Splitting the data into train and test sets
    X = data['text']
    y = data['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Vectorizing the text data using TF-IDF
    vectorizer = TfidfVectorizer(stop_words='english')
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # Training Logistic Regression and Naive Bayes models
    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import MultinomialNB

    lr_model = LogisticRegression()
    lr_model.fit(X_train_vec, y_train)

    nb_model = MultinomialNB()
    nb_model.fit(X_train_vec, y_train)

    # Predictions
    lr_train_predictions = lr_model.predict(X_train_vec)
    lr_test_predictions = lr_model.predict(X_test_vec)

    nb_train_predictions = nb_model.predict(X_train_vec)
    nb_test_predictions = nb_model.predict(X_test_vec)

    # Evaluation Results
    lr_train_accuracy = accuracy_score(y_train, lr_train_predictions)
    lr_test_accuracy = accuracy_score(y_test, lr_test_predictions)
    lr_classification_report = classification_report(y_test, lr_test_predictions)

    nb_train_accuracy = accuracy_score(y_train, nb_train_predictions)
    nb_test_accuracy = accuracy_score(y_test, nb_test_predictions)
    nb_classification_report = classification_report(y_test, nb_test_predictions)

    # Displaying Evaluation Results
    st.write("#### Logistic Regression Evaluation")
    st.text_area("Logistic Regression Classification Report", lr_classification_report, height=200)

    st.write("#### Naive Bayes Evaluation")
    st.text_area("Naive Bayes Classification Report", nb_classification_report, height=200)

    # Confusion Matrices
    lr_cm = confusion_matrix(y_test, lr_test_predictions)
    nb_cm = confusion_matrix(y_test, nb_test_predictions)

    # Plotting confusion matrices for both models
    st.write("#### Confusion Matrices")

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Logistic Regression Confusion Matrix
    sns.heatmap(lr_cm, annot=True, fmt='d', cmap='Blues', ax=axes[0])
    axes[0].set_title('Logistic Regression Confusion Matrix')
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('Actual')

    # Naive Bayes Confusion Matrix
    sns.heatmap(nb_cm, annot=True, fmt='d', cmap='Greens', ax=axes[1])
    axes[1].set_title('Naive Bayes Confusion Matrix')
    axes[1].set_xlabel('Predicted')
    axes[1].set_ylabel('Actual')

    plt.tight_layout()
    st.pyplot(fig)

    # Model Comparison Table
    st.write("### Model Comparison Metrics")
    metrics_data = {
        "Model": ["Logistic Regression", "Naive Bayes"],
        "Training Accuracy": [lr_train_accuracy, nb_train_accuracy],
        "Testing Accuracy": [lr_test_accuracy, nb_test_accuracy],
    }
    metrics_df = pd.DataFrame(metrics_data)
    st.write(metrics_df)

    # Predicting New Articles
    def preprocess_text(text):
        text = re.sub(r'[^a-zA-Z]', ' ', text).lower().strip()
        return text
    
    def predict_news(article):
        processed_text = preprocess_text(article)
        input_vector = vectorizer.transform([processed_text])
        lr_pred = "Fake" if lr_model.predict(input_vector) == 0 else "True"
        nb_pred = "Fake" if nb_model.predict(input_vector) == 0 else "True"
        return lr_pred, nb_pred

    article = st.text_area("Enter a news article:")

    if st.button("Predict"):
        if article.strip():
            lr_prediction, nb_prediction = predict_news(article)
            st.success(f"Logistic Regression Prediction: {lr_prediction}")
            st.success(f"Naive Bayes Prediction: {nb_prediction}")
        else:
            st.warning("Please enter some text to analyze.")

elif section == "Conclusion":
    st.header("Conclusion")
    st.write("Logistic Regression outperforms Naive Bayes in both accuracy and performance metrics.")
    st.write("Logistic Regression achieved higher training (99.16%) and testing accuracy (98.08%) compared to Naive Bayes (94.35% and 93.62%).")
    st.write("Logistic Regression shows better precision, recall, and F1-score, while Naive Bayes is faster but less accurate.")
    st.write("The dataset, limited to USA-based news articles, may affect the model's generalizability to other regions.")

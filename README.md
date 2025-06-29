# 📰 Fake News Detection Web App

This project is a **Streamlit web application** that detects whether a news article is **fake or real** using machine learning models trained on a labeled dataset of real and fake news articles.

## 🚀 Live Demo
[Click here to try the app](https://detector4fakenews.streamlit.app/)  


---

## 📂 Dataset

- **Source**: [Kaggle - Fake and Real News Dataset](https://www.kaggle.com/clmentbisaillon/fake-and-real-news)
- Two files:
  - `True.csv` – 21,417 real news articles
  - `Fake.csv` – 23,502 fake news articles
- Each file contains:
  - `title`: Title of the article
  - `text`: Full body of the news article
  - `subject`: Category of the news
  - `date`: Publish date

---

## 🧠 Models Used

The following machine learning models were trained using **TF-IDF vectorization**:
- **Logistic Regression**
- **Multinomial Naive Bayes**

Model evaluation included:
- Accuracy
- Classification report (Precision, Recall, F1-score)
- Confusion Matrix

---

## 📊 Features

- Paste or upload a news article to check if it’s Fake or Real.
- Choose between two trained models: **Logistic Regression** and **Naive Bayes**.
- Real-time prediction result with clean UI.
- Trained on 45,000+ real and fake news articles.
- TF-IDF Vectorization for feature extraction.


---

## 🛠 How to Run Locally

### 1. Clone the Repository

```bash
git clone https://github.com/itsaimankhan/FakeNewsDetector.git
cd FakeNewsDetection
```
### 2. Install Dependencies
Make sure you have Python 3.7+ installed, then run:

```bash
pip install -r requirements.txt 
```
### 3. Launch the App
``` bash
streamlit run NewsDetectionStreamlit.py
``` 

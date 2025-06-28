# 📰 Fake News Detection Web App

This project is a **Streamlit web application** that detects whether a news article is **fake or real** using machine learning models trained on a labeled dataset of real and fake news articles.

## 🚀 Live Demo
[Click here to try the app](https://your-streamlit-app-url.streamlit.app)  


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

- **EDA Section**: Visualizations including class distribution, word clouds, and top frequent words.
- **Model Section**:
  - Accuracy comparison between Logistic Regression and Naive Bayes
  - Confusion matrices
  - Input field for custom news text to get predictions from both models
- **Conclusion Section**: Highlights key findings from the analysis

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
Copy
Edit
pip install -r requirements.txt 
```
### 3. Launch the App
``` bash
Copy
Edit
streamlit run NewsDetectionStreamlit.py
``` 

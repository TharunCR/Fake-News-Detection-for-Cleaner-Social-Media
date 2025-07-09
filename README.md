# 📰 Fake News Detection for Cleaner Social Media

This project aims to detect fake news articles and reduce the spread of misinformation on social media platforms. Using natural language processing (NLP) and machine learning models, this system can classify news as **fake** or **real** based on the content of the text.

---

## 📌 Project Highlights

- Uses real-world datasets (`Fake.csv` and `True.csv`)
- Cleans and preprocesses textual data (lowercase, punctuation removal, stopword filtering)
- Visualizes frequent words in fake and real news
- Trains multiple ML models: Naive Bayes, Logistic Regression, Decision Tree, and Random Forest
- Evaluates model performance using accuracy and confusion matrix
- Integrates with Hugging Face API (BART model) for double-checking predictions
- Allows real-time user input for prediction via terminal

---

## 🛠️ Technologies Used

| Area               | Tools/Libraries                                |
|--------------------|-------------------------------------------------|
| Programming Language | Python 3.x                                   |
| NLP                 | NLTK, Scikit-learn, CountVectorizer, TF-IDF    |
| Visualization       | Matplotlib, Seaborn, WordCloud                 |
| Models              | Naive Bayes, Logistic Regression, Decision Tree, Random Forest |
| External API        | Hugging Face Transformers API (facebook/bart-large-mnli) |

---

## 📂 Dataset

The project uses two CSV files:
- `Fake.csv`: Contains fake news articles
- `True.csv`: Contains real news articles

Each file includes columns like `title`, `text`, `subject`, and `date`.

---

## 🧹 Data Preprocessing

- Combine `Fake.csv` and `True.csv` into one dataset
- Drop unused columns: `title`, `date`
- Convert text to lowercase
- Remove punctuation
- Remove English stopwords (NLTK)
- Shuffle and reset dataset index

---

## 📊 Exploratory Data Analysis

- Visualizes the distribution of fake vs real news
- Plots most frequent words in fake and real articles
- Uses horizontal bar charts for better readability

---

## 🤖 Model Training & Evaluation

Trained multiple models using **Scikit-learn Pipelines**:

| Model              | Accuracy (%) |
|-------------------|--------------|
| Naive Bayes        | ~95.27%     |
| Logistic Regression| ~98.84%     |
| Decision Tree      | ~99.58%     |
| Random Forest      | ~99.30%     |

- Accuracy and confusion matrix are shown after each model
- The best-performing model is Logistic Regression

---

## 🌐 API Integration

Uses [Hugging Face's](https://huggingface.co/models/facebook/bart-mnli) `facebook/bart-large-mnli` model via API to verify predictions for **fake news**. If the Logistic Regression model predicts "fake", the text is sent to the API to revalidate the result.

> ⚠️ You must use your own Hugging Face API key to run this feature.

---

## 🎯 Real-Time Prediction

Users can interactively input text in the terminal to classify it as **True** or **False**. The system uses the trained Logistic Regression model and validates suspicious content with Hugging Face API.

Enter a sentence to classify (type 'exit' to stop): This is the latest news about the economy...
Final Label: True

📁 Project Structure
graphql
Copy
Edit
FakeNews-Detection/
├── Fake.csv
├── True.csv
├── fake_news_detection.py      # Your full script
├── requirements.txt            # Required libraries
└── README.md

Installation & Setup
Clone the repo
git clone https://github.com/your-username/fake-news-detection.git
cd fake-news-detection


📈 Future Improvements
Integrate deep learning models (e.g., LSTM, BERT)

Build a web interface using Flask or Streamlit

Add language support for multilingual fake news detection

Store user inputs and predictions in a database

👨‍💻 Author
Tharun C R
B.Tech CSE (AI & ML) | VIT-AP

# 🎭 SarcasmIQ - AI-Powered Sarcasm Detection Engine

SarcasmIQ is an NLP-powered web application that detects sarcasm in news headlines and short text using both Classical Machine Learning and Deep Learning approaches. The application provides an interactive Streamlit interface for real-time sarcasm prediction, confidence scoring, and model comparison.

## 🚀 Features

* Real-time sarcasm detection
* Multiple model support

  * Classical ML (TF-IDF + Ensemble Model)
  * Deep Learning (GloVe + BiLSTM)
* Interactive Streamlit dashboard
* Confidence score visualization
* Batch prediction support
* Model performance insights
* Responsive and modern UI

## 🛠️ Tech Stack

### Programming Language

* Python

### Libraries & Frameworks

* Streamlit
* Pandas
* NumPy
* Scikit-learn
* TensorFlow / Keras
* NLTK
* Matplotlib
* Seaborn
* Plotly
* Joblib

### NLP Techniques

* Text Cleaning
* Tokenization
* Stopword Removal
* TF-IDF Vectorization
* Word Embeddings (GloVe)

### Machine Learning Models

* Logistic Regression
* Random Forest
* Gradient Boosting
* Stacking Ensemble

### Deep Learning Models

* Bidirectional LSTM (BiLSTM)

## 📂 Project Structure

```text
SarcasmIQ/
│
├── app.py
├── requirements.txt
├── models/
│   ├── classical_model.pkl
│   ├── tfidf_vectorizer.pkl
│   └── bilstm_model.h5
│
├── data/
│   └── Sarcasm_Headlines_Dataset.json
│
├── assets/
│   └── logo.png
│
└── README.md
```

## 📊 Dataset

The project uses the News Headlines Dataset for Sarcasm Detection containing sarcastic and non-sarcastic news headlines.

Target Labels:

* 1 → Sarcastic
* 0 → Not Sarcastic

## ▶️ Installation

Clone the repository:

```bash
git clone https://github.com/your-username/SarcasmIQ.git
cd SarcasmIQ
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

## 📈 Model Performance

| Model                        | Accuracy |
| ---------------------------- | -------- |
| TF-IDF + Logistic Regression | 78%      |
| TF-IDF + Ensemble            | 80.5%    |
| GloVe + BiLSTM               | 88.9%    |

## 💡 Example

Input:

```text
Scientists discover water is, in fact, wet
```

Output:

```text
Sarcastic
Confidence: 84.8%
```

## 🎯 Future Enhancements

* Transformer-based models (BERT/RoBERTa)
* Explainable AI (SHAP/LIME)
* API deployment with FastAPI
* Multi-language sarcasm detection
* Real-time social media sarcasm analysis

## 👩‍💻 Author

Simran Rani

Data Analyst | Aspiring Data Scientist

Skills: Python, SQL, Machine Learning, Deep Learning, NLP, Streamlit

## 📜 License

This project is intended for educational and portfolio purposes.

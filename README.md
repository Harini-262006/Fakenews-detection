# 📰 Fake News Detection using NLP and Machine Learning

This project is a web-based Fake News Detection system that uses Natural Language Processing (NLP) and a machine learning model to classify news articles as **real** or **fake**. It allows users to input a news headline or article and get an instant prediction.

## 🚀 Features

- 🧠 Trained using a supervised machine learning model
- 🔍 Text preprocessing using NLP techniques (stopword removal, tokenization, etc.)
- 🌐 Flask web interface for user interaction
- 📈 Trained on a real-world dataset of news articles
- 💬 Predicts in real-time whether news is fake or real

## 🛠️ Technologies Used

- Python
- Scikit-learn
- Pandas, NumPy
- Natural Language Toolkit (NLTK)
- Flask
- HTML, CSS (for frontend)
- Git & GitHub
- Git LFS (for handling large dataset files)

## 📂 Project Structure

- FakeNewsDetection/
- │
- ├── model/ # Saved machine learning model (Pickle file)
- ├── static/ # Static files (e.g., CSS)
- ├── templates/ # HTML templates
- ├── app.py # Flask application
- ├── train_model.py # Code for training the model
- ├── news.csv # Dataset (stored via Git LFS if >25MB)
- └── README.md # Project documentation


## 📌 How to Run the Project

1. **Clone the repository**:
   
   git clone https://github.com/Harini-262006/FakeNewsDetection.git
   cd FakeNewsDetection

2. **Train the model (if needed)**:

   python train_model.py

 3. **Run the web application**:

    python app.py

4. **Open in browser** :

  Visit http://127.0.0.1:5000/

 ##  📝 Dataset
- The dataset used is news.csv, which contains labeled real and fake news articles.

- If the dataset is larger than 25MB, it's tracked using Git LFS.

 ## 📊 Model
- TF-IDF Vectorization + PassiveAggressiveClassifier

- Accuracy: ~92% on validation data



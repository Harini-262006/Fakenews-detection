<<<<<<< HEAD
from flask import Flask, render_template, request
import joblib
import string

app = Flask(__name__)

# Load model and vectorizer
model = joblib.load("model/fake_news_model.pkl")
vectorizer = joblib.load("model/tfidf_vectorizer.pkl")

# Text cleaning function
def clean_text(text):
    text = text.lower()
    text = "".join(char for char in text if char not in string.punctuation)
    return text

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    news = request.form.get("news", "")
    if not news.strip():
        return render_template("index.html", prediction="Please enter some news content.", input_news=news)

    cleaned = clean_text(news)
    vectorized = vectorizer.transform([cleaned])
    prediction = model.predict(vectorized)[0]

    # Display 'REAL' or 'FAKE' clearly
    result = "REAL News ✅" if prediction.lower() == "real" else "FAKE News ❌"
    return render_template("index.html", prediction=result, input_news=news)

if __name__ == "__main__":
    app.run(debug=True)
=======
from flask import Flask, render_template, request
import joblib
import string

app = Flask(__name__)

# Load model and vectorizer
model = joblib.load("model/fake_news_model.pkl")
vectorizer = joblib.load("model/tfidf_vectorizer.pkl")

# Text cleaning function
def clean_text(text):
    text = text.lower()
    text = "".join(char for char in text if char not in string.punctuation)
    return text

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    news = request.form.get("news", "")
    if not news.strip():
        return render_template("index.html", prediction="Please enter some news content.", input_news=news)

    cleaned = clean_text(news)
    vectorized = vectorizer.transform([cleaned])
    prediction = model.predict(vectorized)[0]

    # Display 'REAL' or 'FAKE' clearly
    result = "REAL News ✅" if prediction.lower() == "real" else "FAKE News ❌"
    return render_template("index.html", prediction=result, input_news=news)

if __name__ == "__main__":
    app.run(debug=True)
>>>>>>> 5b5a7876322b795fd22c341d22aebfd447a9094e

from flask import Flask, render_template,request
import joblib,re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

def textCleaning(text):
    text = text.lower()
    text =  re.sub(r'\S*[\\/]\S*', "", text)
    text = re.sub(r"[^a-z\s]","",text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text

def wordTokenization(text):
    token = word_tokenize(text)
    stop_words = set(stopwords.words("english"))
    tokenize_text = [word for word in token if word not in stop_words]
    return tokenize_text

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods = ["POST"]) 
def predict():
    news = request.form['news']
    clean_news = textCleaning(news)
    tokenize_news = wordTokenization(clean_news)
    join_news = " ".join(tokenize_news)
    numeric_news = vectorizer.transform([join_news])
    prediction = model.predict(numeric_news)[0]
    result = "Real" if prediction == 1 else "Fake"
    return render_template("index.html", prediction=result)

if __name__ == "__main__":
    app.run(debug=True)

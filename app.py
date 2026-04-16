from flask import Flask, request, render_template
import pickle
import re
import string
import os

app = Flask(__name__)

# Load model safely
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = pickle.load(open(os.path.join(BASE_DIR, "model.pkl"), "rb"))
vectorizer = pickle.load(open(os.path.join(BASE_DIR, "vectorizer.pkl"), "rb"))

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\W', ' ', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    return text

@app.route("/", methods=["GET", "POST"])
def home():
    result = ""
    confidence = ""
    label = ""

    if request.method == "POST":
        news = request.form.get("news")

        if not news or news.strip() == "":
            result = "⚠ Please enter some text"
        else:
            cleaned = clean_text(news)
            vector = vectorizer.transform([cleaned])

            prediction = model.predict(vector)
            proba = model.predict_proba(vector)

            conf = round(max(proba[0]) * 100, 2)

            if prediction[0] == 1:
                result = "✅ Real News"
                label = "real"
            else:
                result = "❌ Fake News"
                label = "fake"

            confidence = f"Confidence: {conf}%"

    return render_template("index.html", result=result, confidence=confidence, label=label)


# IMPORTANT for Render
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
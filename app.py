from flask import Flask, request, render_template
import pickle
import re
import string
import os

app = Flask(__name__)

# ✅ Correct path handling (IMPORTANT for Render)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ✅ Load model safely
try:
    model = pickle.load(open(os.path.join(BASE_DIR, "model.pkl"), "rb"))
    vectorizer = pickle.load(open(os.path.join(BASE_DIR, "vectorizer.pkl"), "rb"))
    print("Model loaded successfully ✅")
except Exception as e:
    print("Error loading model ❌:", e)


# ✅ Text cleaning
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\W', ' ', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    return text


# ✅ Main route
@app.route("/", methods=["GET", "POST"])
def home():
    result = ""
    confidence = ""
    label = ""

    print("Route accessed")  # debug log

    if request.method == "POST":
        news = request.form.get("news")

        if not news or news.strip() == "":
            result = "⚠ Please enter some text"
        else:
            try:
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

            except Exception as e:
                result = "⚠ Error processing input"
                print("Prediction error:", e)

    return render_template("index.html", result=result, confidence=confidence, label=label)


# ✅ Required for Render (but Gunicorn will handle this)
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

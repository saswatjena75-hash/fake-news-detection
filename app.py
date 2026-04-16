from flask import Flask, request, render_template
import pickle
import re
import string
import os

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = None
vectorizer = None

# Load model + vectorizer
def load_model():
    global model, vectorizer
    try:
        model = pickle.load(open(os.path.join(BASE_DIR, "model.pkl"), "rb"))
        vectorizer = pickle.load(open(os.path.join(BASE_DIR, "vectorizer.pkl"), "rb"))
        print("Model loaded successfully ✅")
        print("Model classes:", model.classes_)  # DEBUG
    except Exception as e:
        print("Error loading model ❌:", e)

load_model()

# Clean text
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


@app.route("/", methods=["GET", "POST"])
def home():
    if model is None or vectorizer is None:
        return "⚠ Model failed to load. Check logs."

    result = ""
    confidence = ""
    label = ""

    if request.method == "POST":
        news = request.form.get("news")

        if not news or news.strip() == "":
            result = "⚠ Please enter some text"
        else:
            try:
                cleaned = clean_text(news)
                vector = vectorizer.transform([cleaned])

                prediction = model.predict(vector)[0]
                proba = model.predict_proba(vector)[0]

                conf = round(max(proba) * 100, 2)

                # ✅ SAFE LOGIC (NO GUESSING)
                predicted_class = model.classes_[prediction]

                # Normalize label safely
                predicted_str = str(predicted_class).lower()

                if predicted_str in ["fake", "0", "false"]:
                    result = "❌ Fake News"
                    label = "fake"
                else:
                    result = "✅ Real News"
                    label = "real"

                confidence = f"Confidence: {conf}%"

            except Exception as e:
                result = "⚠ Error processing input"
                print("Prediction error:", e)

    return render_template(
        "index.html",
        result=result,
        confidence=confidence,
        label=label
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

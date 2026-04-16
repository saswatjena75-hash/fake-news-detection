from flask import Flask, request, render_template
import pickle
import re
import os

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load model + vectorizer
model = pickle.load(open(os.path.join(BASE_DIR, "model.pkl"), "rb"))
vectorizer = pickle.load(open(os.path.join(BASE_DIR, "vectorizer.pkl"), "rb"))

print("Model loaded successfully ✅")
print("Model classes:", model.classes_)


# Clean text
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


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
            try:
                # preprocess
                cleaned = clean_text(news)
                vector = vectorizer.transform([cleaned])

                # ✅ YOUR EXACT PREDICTION LOGIC
                prediction = model.predict(vector)[0]
                proba = model.predict_proba(vector)[0]

                conf = round(max(proba) * 100, 2)

                # 🔥 DIRECT FIX (NO GUESSING)
                if str(prediction).lower() in ["1", "true", "real"]:
                    result = "✅ Real News"
                    label = "real"
                else:
                    result = "❌ Fake News"
                    label = "fake"

                confidence = f"Confidence: {conf}%"

            except Exception as e:
                print("Prediction error:", e)
                result = "⚠ Error processing input"

    return render_template(
        "index.html",
        result=result,
        confidence=confidence,
        label=label
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

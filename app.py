from flask import Flask, render_template, request, jsonify
import pickle
import os

app = Flask(__name__)

model = None
vectorizer = None

try:
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)

    with open("vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)

    print("Model and vectorizer loaded successfully")

except Exception as e:
    print("Error loading model files:", e)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        if model is None or vectorizer is None:
            return jsonify({"error": "Model files not loaded properly"})

        data = request.get_json()
        text = data.get("text", "").strip()

        if not text:
            return jsonify({"error": "Please enter some text"})

        text_vector = vectorizer.transform([text])
        prediction = model.predict(text_vector)[0]
        probability = model.predict_proba(text_vector)[0]

        sentiment = "Positive" if prediction == 1 else "Negative"
        confidence = round(max(probability) * 100, 2)

        return jsonify({
            "sentiment": sentiment,
            "confidence": confidence
        })

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
from flask import Flask, render_template, request, jsonify
import pickle

app = Flask(__name__)

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
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
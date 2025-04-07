from flask import Flask, request, jsonify
import onnxruntime
import numpy as np
from transformers import RobertaTokenizer

app = Flask(__name__)

# 🔹 Load tokenizer and ONNX model
print("🔄 Loading tokenizer and ONNX model...")
tokenizer = RobertaTokenizer.from_pretrained("distilroberta-base")
session = onnxruntime.InferenceSession("distilroberta-sequence-classification.onnx")
print("✅ Model is ready!")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        text = data.get("text", "")
        if not text:
            return jsonify({"error": "No text provided"}), 400

        tokens = tokenizer(text, return_tensors="pt", padding="max_length", max_length=128, truncation=True)
        input_ids = tokens["input_ids"].numpy()
        attention_mask = tokens["attention_mask"].numpy()

        inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask
        }

        outputs = session.run(None, inputs)
        prediction = int(np.argmax(outputs[0]))
        label = "positive" if prediction == 1 else "negative"

        return jsonify({"prediction": label})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)

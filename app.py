"""
app.py
------
API Flask avec 3 routes :
  GET  /            → Interface web (index.html)
  POST /predict     → Prédiction Churn
  POST /segment     → Segmentation KMeans
  POST /regression  → Prédiction Monetary
"""

from flask import Flask, request, jsonify, render_template
import sys, os

# Ajouter le dossier racine au path pour trouver src/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.predict import predict, segment, predict_monetary

app = Flask(__name__, template_folder="templates")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict_api():
    """
    Attend : {"data": [recency, frequency, monetary]}
    Retourne : {"churn": 0 ou 1, "probability": 0.xx, "label": "..."}
    """
    try:
        data   = request.json["data"]
        result = predict(data)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/segment", methods=["POST"])
def segment_api():
    """
    Attend : {"data": [recency, frequency, monetary]}
    Retourne : {"segment": 0-3, "label": "..."}
    """
    try:
        data   = request.json["data"]
        result = segment(data)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/regression", methods=["POST"])
def regression_api():
    """
    Attend : {"data": [recency, frequency, monetary]}
    Retourne : {"predicted_monetary": xxx.xx, "label": "..."}
    """
    try:
        data   = request.json["data"]
        result = predict_monetary(data)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)

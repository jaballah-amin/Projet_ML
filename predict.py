"""
predict.py
----------
Fonctions de prédiction utilisées par l'API Flask.
Charge les modèles sauvegardés et applique la même
chaîne de transformation (scaler → PCA) que pendant l'entraînement.

Fix appliqué : utilisation de chemins ABSOLUS basés sur __file__
→ le projet fonctionne peu importe depuis où Flask est lancé.
"""

import joblib
import numpy as np
import os

# ── Chemin racine du projet (2 niveaux au-dessus de src/predict.py) ─────────
# src/ → projet_ml_retail/ → chemin absolu stable
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ── Chemins ABSOLUS des modèles ──────────────────────────────────────────────
MODEL_PATH     = os.path.join(BASE_DIR, "models", "model.pkl")
KMEANS_PATH    = os.path.join(BASE_DIR, "models", "kmeans.pkl")
SCALER_PATH    = os.path.join(BASE_DIR, "models", "scaler.pkl")
PCA_PATH       = os.path.join(BASE_DIR, "models", "pca.pkl")
REGRESSOR_PATH = os.path.join(BASE_DIR, "models", "regressor.pkl")

# ── Noms des segments ────────────────────────────────────────────────────────
SEGMENT_LABELS = {
    0: "💎 Champion — Client VIP fidèle",
    1: "😴 Inactif — Client à risque de départ",
    2: "🆕 Nouveau client — Potentiel à développer",
    3: "🔄 Régulier — Client stable"
}


def _check_models():
    """Vérifie que tous les fichiers .pkl existent avant de les charger."""
    for path in [MODEL_PATH, KMEANS_PATH, SCALER_PATH, PCA_PATH, REGRESSOR_PATH]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"❌ Fichier manquant : {path}\n"
                "   → Lance d'abord : python src/preprocessing.py\n"
                "   → puis          : python src/train_model.py"
            )


def _transform(data):
    """
    Applique la chaîne de transformation :
      raw data [Recency, Frequency, Monetary]
        → StandardScaler
        → PCA (2 composantes)
    """
    scaler = joblib.load(SCALER_PATH)
    pca    = joblib.load(PCA_PATH)

    data_scaled = scaler.transform([data])
    data_pca    = pca.transform(data_scaled)
    return data_pca, data_scaled


def predict(data):
    """
    Prédit si un client va churner (0 = fidèle, 1 = à risque).

    Paramètre :
        data : liste [Recency, Frequency, Monetary]

    Retourne :
        dict avec 'churn' (int), 'probability' (float), 'label' (str)
    """
    _check_models()
    model = joblib.load(MODEL_PATH)
    data_pca, _ = _transform(data)

    churn_pred = int(model.predict(data_pca)[0])
    proba      = model.predict_proba(data_pca)[0]

    return {
        "churn"      : churn_pred,
        "probability": round(float(proba[1]), 4),   # probabilité de churn
        "label"      : "⚠️ Client à risque de Churn" if churn_pred == 1
                        else "✅ Client fidèle"
    }


def segment(data):
    """
    Attribue un segment au client via KMeans.

    Paramètre :
        data : liste [Recency, Frequency, Monetary]

    Retourne :
        dict avec 'segment' (int) et 'label' (str)
    """
    _check_models()
    kmeans = joblib.load(KMEANS_PATH)
    data_pca, _ = _transform(data)

    seg = int(kmeans.predict(data_pca)[0])
    return {
        "segment": seg,
        "label"  : SEGMENT_LABELS.get(seg, f"Segment {seg}")
    }


def predict_monetary(data):
    """
    Prédit la valeur monétaire estimée d'un client.

    Paramètre :
        data : liste [Recency, Frequency, Monetary]
               (Monetary sera ignoré — c'est la target)

    Retourne :
        dict avec 'predicted_monetary' (float) et 'label' (str)
    """
    _check_models()
    regressor = joblib.load(REGRESSOR_PATH)
    scaler    = joblib.load(SCALER_PATH)

    # On scale et on prend uniquement Recency et Frequency (indices 0 et 1)
    data_scaled = scaler.transform([data])
    features    = data_scaled[:, :2]    # colonnes 0=Recency, 1=Frequency

    predicted = float(regressor.predict(features)[0])
    return {
        "predicted_monetary": round(predicted, 2),
        "label": f"Valeur client estimée : £ {predicted:,.2f}"
    }

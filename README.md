# 🛒 Churn Prediction & Customer Segmentation
### Projet Machine Learning — E-Commerce Retail 

---

## 📌 Objectif du projet

Ce projet a pour but de :
1. **Prédire le churn client** — identifier les clients qui risquent de ne plus acheter
2. **Segmenter les clients** — regrouper les clients par profil comportemental via clustering
3. **Estimer la valeur client** — prédire le montant dépensé via régression

Le tout est exposé via une **API Flask** avec une interface web.

---

## 📂 Dataset

**Source** : [Kaggle — E-Commerce Data (Carrie1)](https://www.kaggle.com/datasets/carrie1/ecommerce-data)  
**Fichier à placer** : `data/raw/data.csv` (encodage `latin-1`)

| Colonne | Description |
|---------|-------------|
| `InvoiceNo` | Numéro de facture |
| `StockCode` | Code produit |
| `Description` | Nom du produit |
| `Quantity` | Quantité achetée |
| `InvoiceDate` | Date et heure |
| `UnitPrice` | Prix unitaire (£) |
| `CustomerID` | Identifiant client |
| `Country` | Pays du client |

---

## 🗂️ Structure du projet

```
projet_ml_retail/
├── app/
│   ├── app.py                   # API Flask
│   └── templates/
│       └── index.html           # Interface web
├── src/
│   ├── preprocessing.py         # Nettoyage + RFM + Scaler + PCA
│   ├── train_model.py           # RandomForest + KMeans + Régression + GridSearch
│   ├── predict.py               # Fonctions de prédiction
│   └── utils.py                 # Fonctions utilitaires
├── data/
│   ├── raw/
│   │   └── data.csv             # Dataset brut (à télécharger)
│   ├── processed/               # Données nettoyées
│   └── train_test/              # X_train, X_test, y_train, y_test
├── models/
│   ├── model.pkl                # RandomForest entraîné
│   ├── kmeans.pkl               # KMeans entraîné
│   ├── regressor.pkl            # Régression linéaire
│   ├── scaler.pkl               # StandardScaler
│   ├── pca.pkl                  # PCA
│   └── imputer.pkl              # Imputer
├── reports/
│   ├── correlation_heatmap.png  # Heatmap corrélation
│   ├── elbow_method.png         # Courbe du coude KMeans
│   ├── confusion_matrix.png     # Matrice de confusion
│   └── regression_scatter.png  # Régression réel vs prédit
├── notebooks/
│   └── churn_notebook.ipynb    # Exploration complète
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 🚀 Installation et lancement

### Étape 1 — Créer l'environnement virtuel
```bash
# Créer l'environnement
python -m venv venv

# Activer (Windows)
venv\Scripts\activate

# Activer (Mac/Linux)
source venv/bin/activate
```

### Étape 2 — Installer les dépendances
```bash
pip install -r requirements.txt
```

### Étape 3 — Télécharger le dataset
- Va sur : https://www.kaggle.com/datasets/carrie1/ecommerce-data
- Télécharge `data.csv`
- Place-le dans : `data/raw/data.csv`

### Étape 4 — Lancer le preprocessing
```bash
python src/preprocessing.py
```
Génère les fichiers nettoyés, le scaler, la PCA, et les splits train/test.

### Étape 5 — Entraîner les modèles
```bash
python src/train_model.py
```
Entraîne RandomForest (+ GridSearchCV), KMeans, et la régression linéaire.

### Étape 6 — Lancer l'API Flask
```bash
python app/app.py
```

### Étape 7 — Ouvrir l'interface
```
http://127.0.0.1:5000
```

---

## 🤖 Modèles utilisés

| Modèle | Type | Objectif |
|--------|------|----------|
| Random Forest + GridSearchCV | Classification | Prédire Churn (0/1) |
| KMeans (k=4, Elbow Method) | Clustering | Segmenter les clients |
| Régression Linéaire | Régression | Estimer la valeur client (Monetary) |

---

## 🧪 Routes API

| Route | Méthode | Description |
|-------|---------|-------------|
| `/` | GET | Interface web |
| `/predict` | POST | Prédiction Churn |
| `/segment` | POST | Segmentation client |
| `/regression` | POST | Estimation valeur client |

**Format d'entrée :**
```json
{"data": [recency, frequency, monetary]}
```

**Exemples via PowerShell :**
```powershell
# Churn
Invoke-RestMethod -Uri "http://127.0.0.1:5000/predict" `
  -Method POST -Headers @{"Content-Type"="application/json"} `
  -Body '{"data": [5, 12, 1500]}'

# Segmentation
Invoke-RestMethod -Uri "http://127.0.0.1:5000/segment" `
  -Method POST -Headers @{"Content-Type"="application/json"} `
  -Body '{"data": [5, 12, 1500]}'

# Régression
Invoke-RestMethod -Uri "http://127.0.0.1:5000/regression" `
  -Method POST -Headers @{"Content-Type"="application/json"} `
  -Body '{"data": [5, 12, 1500]}'
```

---

## 📦 Dépendances

```
pandas, numpy, scikit-learn, flask, joblib, matplotlib, seaborn
```

---

## 👤 Auteur

Projet réalisé par ** JABALLAH Mohamed Amin ** —  GI2s1 (2025-2026)

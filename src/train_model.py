"""
train_model.py
--------------
Entraînement de 3 modèles :
  1. Random Forest Classifier  → prédire Churn (0/1)
  2. KMeans Clustering         → segmenter les clients (4 groupes)
  3. Linear Regression         → prédire Monetary (valeur client)

Pour chaque modèle supervisé : GridSearchCV pour trouver les meilleurs hyperparamètres.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model  import LinearRegression
from sklearn.cluster       import KMeans
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    mean_squared_error, r2_score
)
import joblib
import os

os.makedirs("reports", exist_ok=True)
os.makedirs("models",  exist_ok=True)


# ─────────────────────────────────────────────
# Chargement des données
# ─────────────────────────────────────────────
def load_data():
    X_train = pd.read_csv("data/train_test/X_train.csv")
    X_test  = pd.read_csv("data/train_test/X_test.csv")
    y_train = pd.read_csv("data/train_test/y_train.csv").values.ravel()
    y_test  = pd.read_csv("data/train_test/y_test.csv").values.ravel()

    # Pour la régression (données scalées sans PCA)
    X_train_sc = pd.read_csv("data/train_test/X_train_scaled.csv")
    X_test_sc  = pd.read_csv("data/train_test/X_test_scaled.csv")
    y_train_reg = pd.read_csv("data/train_test/y_train_reg.csv").values.ravel()
    y_test_reg  = pd.read_csv("data/train_test/y_test_reg.csv").values.ravel()

    return (X_train, X_test, y_train, y_test,
            X_train_sc, X_test_sc, y_train_reg, y_test_reg)


# ─────────────────────────────────────────────
# MODÈLE 1 — Random Forest + GridSearchCV
# ─────────────────────────────────────────────
def train_classifier(X_train, X_test, y_train, y_test):
    """
    GridSearchCV teste toutes les combinaisons d'hyperparamètres
    et sélectionne la meilleure via validation croisée (cv=5).
    """
    print("\n" + "="*55)
    print("  MODÈLE 1 — Random Forest Classifier (Churn)")
    print("="*55)

    # Grille des hyperparamètres à tester
    param_grid = {
        "n_estimators": [50, 100, 200],
        "max_depth"   : [None, 5, 10],
        "class_weight": ["balanced"]   # important car classes potentiellement déséquilibrées
    }

    rf = RandomForestClassifier(random_state=42)

    print("🔍 GridSearchCV en cours (cela peut prendre ~30 sec)...")
    grid_search = GridSearchCV(
        estimator  = rf,
        param_grid = param_grid,
        cv         = 5,           # 5-fold cross-validation
        scoring    = "f1",        # F1 = bon pour le churn (équilibre précision/rappel)
        n_jobs     = -1,          # utilise tous les cœurs CPU
        verbose    = 1
    )
    grid_search.fit(X_train, y_train)

    print(f"\n✅ Meilleurs hyperparamètres : {grid_search.best_params_}")
    print(f"   Meilleur score F1 (CV)   : {grid_search.best_score_:.4f}")

    best_model = grid_search.best_estimator_

    # Évaluation sur le jeu de test
    preds = best_model.predict(X_test)
    print(f"\n📊 Accuracy sur test : {accuracy_score(y_test, preds):.4f}")
    print("\n📋 Classification Report :")
    print(classification_report(y_test, preds,
                                 target_names=["Fidèle (0)", "Churn (1)"]))

    # Matrice de confusion → image
    cm = confusion_matrix(y_test, preds)
    plt.figure(figsize=(5, 4))
    plt.imshow(cm, cmap="Blues")
    plt.title("Matrice de confusion — Random Forest")
    plt.colorbar()
    plt.xlabel("Prédit"); plt.ylabel("Réel")
    plt.xticks([0,1], ["Fidèle","Churn"])
    plt.yticks([0,1], ["Fidèle","Churn"])
    for i in range(2):
        for j in range(2):
            plt.text(j, i, cm[i,j], ha="center", va="center",
                     color="white" if cm[i,j] > cm.max()/2 else "black",
                     fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig("reports/confusion_matrix.png")
    plt.close()
    print("✅ Matrice de confusion → reports/confusion_matrix.png")

    joblib.dump(best_model, "models/model.pkl")
    print("✅ Modèle sauvegardé → models/model.pkl")
    return best_model


# ─────────────────────────────────────────────
# MODÈLE 2 — KMeans Clustering
# ─────────────────────────────────────────────
def train_clustering(X_train):
    """
    Méthode du coude (Elbow Method) pour choisir k optimal.
    On entraîne ensuite KMeans avec ce k.
    """
    print("\n" + "="*55)
    print("  MODÈLE 2 — KMeans Clustering (Segmentation)")
    print("="*55)

    # ─── Elbow Method ───────────────────────────
    inertias = []
    k_range  = range(2, 10)

    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X_train)
        inertias.append(km.inertia_)

    plt.figure(figsize=(7, 4))
    plt.plot(list(k_range), inertias, "bo-", markersize=8)
    plt.xlabel("Nombre de clusters k")
    plt.ylabel("Inertie (SSE)")
    plt.title("Méthode du coude — Choix de k optimal")
    plt.xticks(list(k_range))
    plt.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.savefig("reports/elbow_method.png")
    plt.close()
    print("✅ Graphique Elbow → reports/elbow_method.png")

    # ─── Entraînement avec k=4 ─────────────────
    print("\n🔧 Entraînement KMeans avec k=4 (coude identifié)...")
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    kmeans.fit(X_train)

    labels = kmeans.labels_
    unique, counts = np.unique(labels, return_counts=True)
    print("Distribution des segments :")
    segment_names = {0: "💎 Champions", 1: "😴 Inactifs",
                     2: "🆕 Nouveaux",  3: "🔄 Réguliers"}
    for seg, cnt in zip(unique, counts):
        print(f"  Segment {seg} {segment_names.get(seg,'')} : {cnt} clients")

    joblib.dump(kmeans, "models/kmeans.pkl")
    print("✅ Modèle sauvegardé → models/kmeans.pkl")
    return kmeans


# ─────────────────────────────────────────────
# MODÈLE 3 — Régression Linéaire (Monetary)
# ─────────────────────────────────────────────
def train_regression(X_train_sc, X_test_sc, y_train_reg, y_test_reg):
    """
    Prédit la valeur monétaire (Monetary) d'un client à partir de
    Recency et Frequency → utile pour estimer sa valeur future.
    On utilise seulement Recency et Frequency comme features
    (Monetary est la target, donc on ne peut pas l'utiliser comme feature).
    """
    print("\n" + "="*55)
    print("  MODÈLE 3 — Régression Linéaire (Monetary)")
    print("="*55)

    # On retire Monetary des features (c'est la target !)
    features = ["Recency", "Frequency"]
    X_tr = X_train_sc[features]
    X_te = X_test_sc[features]

    reg = LinearRegression()
    reg.fit(X_tr, y_train_reg)

    preds_reg = reg.predict(X_te)
    mse  = mean_squared_error(y_test_reg, preds_reg)
    rmse = np.sqrt(mse)
    r2   = r2_score(y_test_reg, preds_reg)

    print(f"\n📊 RMSE  : {rmse:.2f} £")
    print(f"   R²    : {r2:.4f}  (1.0 = parfait, 0 = nul)")
    print(f"   Coeff Recency   : {reg.coef_[0]:.4f}")
    print(f"   Coeff Frequency : {reg.coef_[1]:.4f}")
    print(f"   Intercept       : {reg.intercept_:.4f}")

    # Graphique valeurs réelles vs prédites
    plt.figure(figsize=(6, 5))
    plt.scatter(y_test_reg, preds_reg, alpha=0.4, color="steelblue", s=20)
    mn = min(y_test_reg.min(), preds_reg.min())
    mx = max(y_test_reg.max(), preds_reg.max())
    plt.plot([mn, mx], [mn, mx], "r--", lw=2, label="Parfait")
    plt.xlabel("Monetary réel (£)")
    plt.ylabel("Monetary prédit (£)")
    plt.title(f"Régression Linéaire — R² = {r2:.3f}")
    plt.legend()
    plt.tight_layout()
    plt.savefig("reports/regression_scatter.png")
    plt.close()
    print("✅ Graphique régression → reports/regression_scatter.png")

    joblib.dump(reg, "models/regressor.pkl")
    print("✅ Modèle sauvegardé → models/regressor.pkl")
    return reg


# ─────────────────────────────────────────────
# Pipeline principal
# ─────────────────────────────────────────────
def train():
    print("📂 Chargement des données train/test...")
    (X_train, X_test, y_train, y_test,
     X_train_sc, X_test_sc, y_train_reg, y_test_reg) = load_data()

    print(f"X_train shape : {X_train.shape}")
    print(f"X_test  shape : {X_test.shape}")

    train_classifier(X_train, X_test, y_train, y_test)
    train_clustering(X_train)
    train_regression(X_train_sc, X_test_sc, y_train_reg, y_test_reg)

    print("\n🎉 Tous les modèles sont entraînés et sauvegardés !")


if __name__ == "__main__":
    train()

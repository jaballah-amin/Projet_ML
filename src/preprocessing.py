"""
preprocessing.py
----------------
Chaîne complète de préparation des données :
  1. Chargement & nettoyage
  2. Feature Engineering (RFM)
  3. Création du label Churn
  4. Imputation des valeurs manquantes
  5. Détection & correction des outliers
  6. Analyse de corrélation
  7. Split Train / Test  ← AVANT tout fit() du scaler/PCA
  8. StandardScaler  ← fit sur X_train uniquement
  9. PCA            ← fit sur X_train uniquement
  10. Sauvegarde
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")          # pas de fenêtre graphique nécessaire
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import joblib


# ─────────────────────────────────────────────
# 0.  Création des dossiers si absents
# ─────────────────────────────────────────────
for folder in [
    "data/raw", "data/processed", "data/train_test",
    "models", "reports"
]:
    os.makedirs(folder, exist_ok=True)


# ─────────────────────────────────────────────
# 1.  Chargement
# ─────────────────────────────────────────────
def load_raw_data(path="data/raw/data.csv"):
    """Charge le CSV brut avec encodage latin-1 (nécessaire pour ce dataset)."""
    df = pd.read_csv(path, encoding="latin-1")
    print(f"✅ Données chargées : {df.shape[0]} lignes × {df.shape[1]} colonnes")
    return df


# ─────────────────────────────────────────────
# 2.  Nettoyage
# ─────────────────────────────────────────────
def clean_data(df):
    """
    - Supprime les lignes sans CustomerID (on ne peut pas les rattacher à un client)
    - Supprime les retours (Quantity < 0) et les prix invalides (UnitPrice <= 0)
    - Convertit les types
    - Crée la colonne Revenue = Quantity × UnitPrice
    """
    print("\n--- Nettoyage ---")
    print(f"Avant nettoyage : {len(df)} lignes")

    df = df.dropna(subset=["CustomerID"])
    df = df[df["Quantity"] > 0]
    df = df[df["UnitPrice"] > 0]

    df["CustomerID"] = df["CustomerID"].astype(int)
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    df["Revenue"] = df["Quantity"] * df["UnitPrice"]

    print(f"Après nettoyage  : {len(df)} lignes")
    return df


# ─────────────────────────────────────────────
# 3.  Feature Engineering — RFM
# ─────────────────────────────────────────────
def build_rfm(df):
    """
    Agrège toutes les transactions par client pour obtenir 3 features :
      - Recency   : nb de jours depuis le dernier achat (petit = actif)
      - Frequency : nb de commandes distinctes
      - Monetary  : total dépensé (£)
    On passe de ~400 000 lignes à ~4 372 clients uniques.
    """
    print("\n--- Feature Engineering RFM ---")
    snapshot_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)

    rfm = df.groupby("CustomerID").agg(
        Recency  =("InvoiceDate", lambda x: (snapshot_date - x.max()).days),
        Frequency=("InvoiceNo",   "nunique"),
        Monetary =("Revenue",     "sum")
    ).reset_index()

    print(f"Clients uniques après agrégation RFM : {len(rfm)}")
    print(rfm[["Recency", "Frequency", "Monetary"]].describe().round(2))
    return rfm


# ─────────────────────────────────────────────
# 4.  Création du label Churn
# ─────────────────────────────────────────────
def create_churn_label(rfm):
    """
    Le dataset ne contient pas de colonne Churn — on la crée via une règle métier :
      Churn = 1  si Recency > médiane  (client inactif depuis trop longtemps)
      Churn = 0  sinon                 (client encore actif)
    La médiane garantit un équilibre 50/50 entre les deux classes.
    """
    median_recency = rfm["Recency"].median()
    rfm["Churn"] = (rfm["Recency"] > median_recency).astype(int)

    print(f"\n--- Label Churn (seuil = {median_recency:.0f} jours) ---")
    print(rfm["Churn"].value_counts())
    return rfm, median_recency


# ─────────────────────────────────────────────
# 5.  Imputation des valeurs manquantes
# ─────────────────────────────────────────────
def impute_missing(X_train, X_test):
    """
    ⚠️  L'imputation suit la même règle que le scaler :
        - fit sur X_train uniquement
        - transform sur X_train ET X_test
    Stratégie : médiane (robuste aux outliers pour des données financières)
    """
    print("\n--- Imputation ---")
    print(f"Valeurs manquantes avant : {X_train.isnull().sum().sum()}")

    imputer = SimpleImputer(strategy="median")
    X_train_imp = pd.DataFrame(
        imputer.fit_transform(X_train),
        columns=X_train.columns
    )
    X_test_imp = pd.DataFrame(
        imputer.transform(X_test),
        columns=X_test.columns
    )

    print(f"Valeurs manquantes après : {X_train_imp.isnull().sum().sum()}")
    joblib.dump(imputer, "models/imputer.pkl")
    return X_train_imp, X_test_imp


# ─────────────────────────────────────────────
# 6.  Détection & traitement des outliers (IQR)
# ─────────────────────────────────────────────
def remove_outliers_iqr(df, columns):
    """
    Méthode IQR (Interquartile Range) :
      - Q1 = 25e percentile, Q3 = 75e percentile
      - IQR = Q3 - Q1
      - Outlier si valeur < Q1 - 1.5×IQR  ou  > Q3 + 1.5×IQR
    On SUPPRIME les lignes outliers pour garder des données propres.
    """
    print("\n--- Détection outliers (IQR) ---")
    before = len(df)
    for col in columns:
        Q1  = df[col].quantile(0.25)
        Q3  = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        df = df[(df[col] >= lower) & (df[col] <= upper)]
    after = len(df)
    print(f"Lignes supprimées (outliers) : {before - after} ({100*(before-after)/before:.1f}%)")
    return df


# ─────────────────────────────────────────────
# 7.  Analyse de corrélation (heatmap)
# ─────────────────────────────────────────────
def plot_correlation(X, save_path="reports/correlation_heatmap.png"):
    """
    Calcule et sauvegarde la heatmap de corrélation.
    Règle : si |corrélation| > 0.8 entre deux features → considérer suppression.
    """
    print("\n--- Analyse de corrélation ---")
    corr = X.corr()
    print(corr.round(3))

    plt.figure(figsize=(6, 4))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
                square=True, linewidths=0.5)
    plt.title("Matrice de corrélation — Features RFM")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"✅ Heatmap sauvegardée : {save_path}")


# ─────────────────────────────────────────────
# 8.  Pipeline principal
# ─────────────────────────────────────────────
def preprocess(data_path="data/raw/data.csv"):

    # ── Chargement & nettoyage ──────────────────
    df  = load_raw_data(data_path)
    df  = clean_data(df)

    # ── RFM & label ────────────────────────────
    rfm = build_rfm(df)
    rfm, _ = create_churn_label(rfm)

    # ── Suppression outliers ────────────────────
    rfm = remove_outliers_iqr(rfm, ["Recency", "Frequency", "Monetary"])

    # ── Séparation features / target ───────────
    X = rfm[["Recency", "Frequency", "Monetary"]]
    y = rfm["Churn"]

    # ── Analyse corrélation (sur X brut) ───────
    plot_correlation(X)

    # ── Sauvegarde données traitées ─────────────
    rfm.to_csv("data/processed/rfm_cleaned.csv", index=False)
    print("✅ Données nettoyées sauvegardées → data/processed/rfm_cleaned.csv")

    # ─────────────────────────────────────────────
    # ⚠️  RÈGLE IMPORTANTE : Train/Test split AVANT fit() du scaler et PCA
    #     Pourquoi ? Si on fit le scaler sur tout X, les stats du test
    #     "contaminent" l'entraînement → le modèle triche = data leakage.
    # ─────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y          # conserve la proportion 50/50 Churn dans train et test
    )
    print(f"\n--- Split ---")
    print(f"X_train : {X_train.shape}   X_test : {X_test.shape}")

    # ── Imputation ─────────────────────────────
    X_train, X_test = impute_missing(X_train, X_test)

    # ── StandardScaler ─────────────────────────
    # fit UNIQUEMENT sur X_train, transform sur les deux
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)       # ← pas fit_transform !
    print("\n--- StandardScaler appliqué ---")

    # ── PCA ────────────────────────────────────
    # fit UNIQUEMENT sur X_train_scaled
    pca = PCA(n_components=2)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca  = pca.transform(X_test_scaled)      # ← pas fit_transform !
    print(f"Variance expliquée par les 2 composantes PCA : "
          f"{pca.explained_variance_ratio_.sum()*100:.1f}%")

    # ── Sauvegarde train/test ───────────────────
    pd.DataFrame(X_train_pca, columns=["PC1","PC2"]).to_csv("data/train_test/X_train.csv", index=False)
    pd.DataFrame(X_test_pca,  columns=["PC1","PC2"]).to_csv("data/train_test/X_test.csv",  index=False)
    pd.Series(y_train.values, name="Churn").to_csv("data/train_test/y_train.csv", index=False)
    pd.Series(y_test.values,  name="Churn").to_csv("data/train_test/y_test.csv",  index=False)

    # Sauvegarde aussi les données scalées SANS PCA (pour la régression)
    pd.DataFrame(X_train_scaled, columns=["Recency","Frequency","Monetary"]).to_csv(
        "data/train_test/X_train_scaled.csv", index=False)
    pd.DataFrame(X_test_scaled,  columns=["Recency","Frequency","Monetary"]).to_csv(
        "data/train_test/X_test_scaled.csv",  index=False)
    pd.Series(rfm.loc[y_train.index, "Monetary"].values, name="Monetary").to_csv(
        "data/train_test/y_train_reg.csv", index=False)
    pd.Series(rfm.loc[y_test.index,  "Monetary"].values, name="Monetary").to_csv(
        "data/train_test/y_test_reg.csv",  index=False)

    # ── Sauvegarde modèles de transformation ───
    joblib.dump(scaler, "models/scaler.pkl")
    joblib.dump(pca,    "models/pca.pkl")
    print("\n✅ scaler.pkl et pca.pkl sauvegardés dans models/")

    print("\n🎉 preprocessing terminé avec succès !")


if __name__ == "__main__":
    preprocess()

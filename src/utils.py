"""
utils.py
--------
Fonctions utilitaires réutilisables dans tout le projet :
  - Chargement / sauvegarde de données
  - Statistiques descriptives
  - Analyse de corrélation & VIF
  - Détection d'outliers
  - Affichage de distribution
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import os


# ─────────────────────────────────────────────
# I/O
# ─────────────────────────────────────────────

def load_data(path, encoding="utf-8"):
    """Charge un fichier CSV."""
    df = pd.read_csv(path, encoding=encoding)
    print(f"✅ {os.path.basename(path)} chargé : {df.shape}")
    return df


def save_data(df, path):
    """Sauvegarde un DataFrame au format CSV."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"✅ Sauvegardé : {path}")


# ─────────────────────────────────────────────
# Statistiques descriptives
# ─────────────────────────────────────────────

def describe_dataframe(df):
    """
    Affiche un résumé complet du DataFrame :
      - Types, nb de NaN, statistiques, valeurs uniques
    """
    print("\n" + "="*55)
    print("  Résumé du DataFrame")
    print("="*55)
    print(f"Dimensions : {df.shape[0]} lignes × {df.shape[1]} colonnes\n")

    print("Types & valeurs manquantes :")
    missing = df.isnull().sum()
    pct     = (missing / len(df) * 100).round(2)
    info_df = pd.DataFrame({
        "dtype"  : df.dtypes,
        "nan"    : missing,
        "nan_%"  : pct
    })
    print(info_df[info_df["nan"] > 0] if (missing > 0).any() else "Aucune valeur manquante ✅")

    print("\nStatistiques descriptives (numériques) :")
    print(df.describe().round(2))
    return info_df


# ─────────────────────────────────────────────
# Analyse de corrélation
# ─────────────────────────────────────────────

def correlation_analysis(df, threshold=0.8, save_path=None):
    """
    Calcule la matrice de corrélation et identifie les paires
    fortement corrélées (|r| > threshold).

    Règle du PDF : si |corrélation| > 0.8 → envisager de supprimer une feature.
    """
    print("\n--- Analyse de corrélation ---")
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    corr = df[num_cols].corr()

    # Paires fortement corrélées
    high_corr_pairs = []
    for i in range(len(corr.columns)):
        for j in range(i+1, len(corr.columns)):
            val = abs(corr.iloc[i, j])
            if val > threshold:
                high_corr_pairs.append((corr.columns[i], corr.columns[j], round(val, 3)))

    if high_corr_pairs:
        print(f"⚠️  Paires avec |corrélation| > {threshold} (multicolinéarité) :")
        for a, b, v in high_corr_pairs:
            print(f"   {a}  ↔  {b}  :  {v}")
    else:
        print(f"✅ Aucune paire avec |corrélation| > {threshold}")

    if save_path:
        plt.figure(figsize=(max(6, len(num_cols)), max(5, len(num_cols)-1)))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
                    square=True, linewidths=0.5, vmin=-1, vmax=1)
        plt.title("Matrice de corrélation")
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        print(f"✅ Heatmap sauvegardée : {save_path}")

    return corr


# ─────────────────────────────────────────────
# Détection d'outliers
# ─────────────────────────────────────────────

def detect_outliers_iqr(df, columns):
    """
    Détecte les outliers via la méthode IQR pour chaque colonne.
    Retourne un DataFrame avec le nombre d'outliers par colonne.
    """
    print("\n--- Détection d'outliers (IQR) ---")
    results = []
    for col in columns:
        Q1  = df[col].quantile(0.25)
        Q3  = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        n_out = ((df[col] < lower) | (df[col] > upper)).sum()
        results.append({
            "feature" : col,
            "Q1"      : round(Q1,  2),
            "Q3"      : round(Q3,  2),
            "IQR"     : round(IQR, 2),
            "lower"   : round(lower, 2),
            "upper"   : round(upper, 2),
            "outliers": n_out,
            "outliers_%": round(n_out/len(df)*100, 2)
        })
    report = pd.DataFrame(results)
    print(report.to_string(index=False))
    return report


# ─────────────────────────────────────────────
# Visualisation des distributions
# ─────────────────────────────────────────────

def plot_distributions(df, columns, save_path="reports/distributions.png"):
    """
    Génère des histogrammes + boxplots pour les colonnes spécifiées.
    Utile pour visualiser rapidement la forme des distributions.
    """
    n = len(columns)
    fig, axes = plt.subplots(2, n, figsize=(5*n, 8))

    for i, col in enumerate(columns):
        # Histogramme
        axes[0, i].hist(df[col].dropna(), bins=40, color="steelblue",
                        edgecolor="white", alpha=0.8)
        axes[0, i].set_title(f"Distribution — {col}")
        axes[0, i].set_xlabel(col)
        axes[0, i].set_ylabel("Fréquence")

        # Boxplot
        axes[1, i].boxplot(df[col].dropna(), patch_artist=True,
                           boxprops=dict(facecolor="lightblue"))
        axes[1, i].set_title(f"Boxplot — {col}")
        axes[1, i].set_ylabel(col)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    print(f"✅ Distributions sauvegardées : {save_path}")


# ─────────────────────────────────────────────
# Vérification équilibre des classes
# ─────────────────────────────────────────────

def check_class_balance(y, label="Churn"):
    """
    Affiche la distribution d'une variable cible binaire.
    Important pour détecter un déséquilibre de classes.
    """
    counts = pd.Series(y).value_counts()
    pct    = pd.Series(y).value_counts(normalize=True) * 100
    print(f"\n--- Équilibre des classes — {label} ---")
    for cls in counts.index:
        print(f"  Classe {cls} : {counts[cls]} exemples ({pct[cls]:.1f}%)")
    ratio = counts.max() / counts.min()
    if ratio > 2:
        print(f"⚠️  Déséquilibre détecté (ratio {ratio:.1f}x) → utiliser class_weight='balanced'")
    else:
        print(f"✅ Classes équilibrées (ratio {ratio:.1f}x)")

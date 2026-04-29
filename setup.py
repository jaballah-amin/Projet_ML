"""
setup.py
--------
Script à lancer UNE SEULE FOIS pour créer tous les dossiers nécessaires.
Usage : python setup.py
"""
import os

folders = [
    "data/raw", "data/processed", "data/train_test",
    "models", "reports", "notebooks",
    "app/templates", "src"
]

print("📁 Création de la structure du projet...")
for folder in folders:
    os.makedirs(folder, exist_ok=True)
    print(f"   ✅ {folder}/")

print("\n🎉 Structure créée ! Maintenant :")
print("   1. Télécharge data.csv depuis Kaggle → data/raw/data.csv")
print("   2. python src/preprocessing.py")
print("   3. python src/train_model.py")
print("   4. python app/app.py")

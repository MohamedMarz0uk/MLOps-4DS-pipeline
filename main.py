import warnings

warnings.filterwarnings("ignore")

# Importer toutes les fonctions depuis model_pipeline.py
from model_pipeline import (
    prepare_data,
    train_model,
    evaluate_model,
    predict,
    save_model,
    load_model,
)


def main():
    # Chemins vers les fichiers CSV
    train_path = "data/train.csv"
    test_path = "data/test.csv"

    print("=== Pipeline de Prédiction de Prêts - Modèle Logistic Regression ===\n")

    # -----------------------
    # 1. Préparation des données
    # -----------------------
    print("1. Préparation des données...")
    data = prepare_data(train_path, test_path)
    print(f"Shape X_train: {data['X_train'].shape}")
    print(f"Shape X_test: {data['X_test'].shape}")
    print(f"Shape X_final_test: {data['X_final_test'].shape}")
    print(f"Features utilisées: {list(data['X_train'].columns)}\n")

    # -----------------------
    # 2. Entraînement du modèle
    # -----------------------
    print("2. Entraînement du modèle Logistic Regression...")
    model = train_model(data["X_train"], data["y_train"])
    print()

    # -----------------------
    # 3. Évaluation du modèle
    # -----------------------
    print("3. Évaluation du modèle...")
    results = evaluate_model(model, data["X_test"], data["y_test"])
    print()

    # -----------------------
    # 4. Prédiction sur les données de test finales
    # -----------------------
    print("4. Prédictions sur les données de test...")
    final_predictions = predict(model, data["X_final_test"])
    print(f"Nombre de prédictions: {len(final_predictions)}")
    print(f"Prêts approuvés: {sum(final_predictions == 1)}")
    print(f"Prêts refusés: {sum(final_predictions == 0)}\n")

    # -----------------------
    # 5. Sauvegarde du modèle
    # -----------------------
    print("5. Sauvegarde du modèle...")
    save_model(model, "logistic_regression_loan_model.pkl")
    print()

    # -----------------------
    # 6. Test du chargement du modèle
    # -----------------------
    print("6. Test de chargement du modèle...")
    loaded_model = load_model("logistic_regression_loan_model.pkl")
    test_predictions = predict(loaded_model, data["X_test"].head())
    print(f"Prédictions avec le modèle chargé : {test_predictions}\n")

    print("=== Pipeline terminé avec succès ===")


# Correction ici (double underscore !)
if __name__ == "__main__":
    main()

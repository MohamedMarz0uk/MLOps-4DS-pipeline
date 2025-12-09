import warnings
warnings.filterwarnings("ignore")

import os
os.environ["MLFLOW_TRACKING_URI"] = "file:./mlruns"
os.environ["MLFLOW_ARTIFACT_URI"] = "./mlruns/artifacts"

import mlflow
from model_pipeline import (
    prepare_data,
    train_model,
    evaluate_model,
    predict,
    save_model,
    save_model_mlflow,
    load_model_from_mlflow,
    create_submission,
)


def main():
    train_path = "data/train.csv"
    test_path = "data/test.csv"

    print("=== Pipeline de Prédiction de Prêts - Modèle Logistic Regression avec MLflow ===\n")

    #mlflow.set_tracking_uri("sqlite:///mlflow.db")
    
    mlflow.set_experiment("loan_prediction_experiment")
    
    # Démarrer un run MLflow
    with mlflow.start_run(run_name="loan_prediction_baseline") as run:
        print(f"MLflow Run ID: {run.info.run_id}")
        print(f"MLflow Experiment ID: {run.info.experiment_id}\n")
        
        # Log des tags
        mlflow.set_tag("model_type", "LogisticRegression")
        mlflow.set_tag("dataset", "loan_prediction")
        mlflow.set_tag("version", "1.0")
        
        print("1. Préparation des données...")
        data = prepare_data(train_path, test_path)
        print(f"Shape X_train: {data['X_train'].shape}")
        print(f"Shape X_test: {data['X_test'].shape}")
        print(f"Shape X_final_test: {data['X_final_test'].shape}")
        print(f"Features utilisées: {list(data['X_train'].columns)}\n")

        print("2. Entraînement du modèle Logistic Regression...")
        model = train_model(data["X_train"], data["y_train"])
        print()

        print("3. Évaluation du modèle...")
        results = evaluate_model(model, data["X_test"], data["y_test"])
        print()

        print("4. Prédictions sur les données de test...")
        final_predictions = predict(model, data["X_final_test"])
        print(f"Nombre de prédictions: {len(final_predictions)}")
        print(f"Prêts approuvés: {sum(final_predictions == 1)}")
        print(f"Prêts refusés: {sum(final_predictions == 0)}\n")

        print("5. Sauvegarde du modèle dans MLflow...")
        model_uri = save_model_mlflow(model, data["X_train"])
        print(f"Modèle URI: {model_uri}\n")
        
        print("6. Sauvegarde locale (pickle) du modèle...")
        save_model(model, "logistic_regression_loan_model.pkl", metrics=results)
        print()

        print("7. Création du fichier de soumission...")
        submission = create_submission(
            final_predictions, 
            data["test_ids"], 
            "submission.csv"
        )
        print()

        print("8. Test de chargement du modèle depuis MLflow...")
        try:
            loaded_model = load_model_from_mlflow("loan_prediction_model")
            test_predictions = predict(loaded_model, data["X_test"].head())
            print(f"Prédictions avec le modèle chargé : {test_predictions}\n")
        except Exception as e:
            print(f"Note: Chargement depuis MLflow registry non disponible: {e}")
            print("Le modèle est disponible via l'URI du run.\n")

        print("=== Pipeline terminé avec succès ===")
        print(f"\nPour visualiser les résultats dans MLflow UI, exécutez:")
        print("mlflow ui")
        print("Puis ouvrez http://localhost:4000 dans votre navigateur")


def train_with_hyperparameters():
    """
    Exemple d'entraînement avec différents hyperparamètres pour comparaison.
    """
    train_path = "data/train.csv"
    test_path = "data/test.csv"
    
    print("=== Entraînement avec plusieurs hyperparamètres ===\n")
    
    # Préparer les données une seule fois
    data = prepare_data(train_path, test_path)
    
    # Tester différentes valeurs de C
    C_values = [0.01, 0.1, 1.0, 10.0, 100.0]
    
    for C in C_values:
        with mlflow.start_run(run_name=f"loan_prediction_C_{C}"):
            print(f"\n--- Entraînement avec C={C} ---")
            mlflow.set_tag("hyperparameter_tuning", "C_variation")
            
            model = train_model(data["X_train"], data["y_train"], C=C)
            results = evaluate_model(model, data["X_test"], data["y_test"])
            
            print(f"Accuracy: {results['accuracy']:.4f}")
            print(f"F1-Score: {results['f1_score']:.4f}")
    
    print("\n=== Entraînement terminé ===")
    print("Comparez les runs dans MLflow UI pour choisir le meilleur modèle")


if __name__ == "__main__":
    # Exécuter le pipeline principal
    main()
    
    # Décommenter pour tester avec différents hyperparamètres
    # train_with_hyperparameters()
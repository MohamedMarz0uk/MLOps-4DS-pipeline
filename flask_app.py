from flask import Flask, render_template, request
import pandas as pd
from model_pipeline import (
    preprocess_data, 
    load_model, 
    predict, 
    prepare_data, 
    save_model,
    retrain_model,
    evaluate_model
)
from sklearn.metrics import confusion_matrix
import os

app = Flask(__name__)

# Charger le modèle initial
try:
    model = load_model()
    trained_columns = getattr(model, "feature_names_in_", None)
except FileNotFoundError:
    print("⚠️ Aucun modèle trouvé. Veuillez entraîner un modèle d'abord avec main.py")
    model = None
    trained_columns = None


# ----------------------------
# Page d'accueil & Prédiction
# ----------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    global model, trained_columns

    if model is None:
        return "⚠️ Modèle non chargé. Veuillez entraîner le modèle d'abord.", 503

    if request.method == "POST":
        # Collecter les données du formulaire
        form_data = request.form.to_dict()
        df = pd.DataFrame([form_data])

        # Convertir les champs numériques
        numeric_fields = [
            "ApplicantIncome",
            "CoapplicantIncome", 
            "LoanAmount",
            "Loan_Amount_Term",
            "Credit_History"
        ]
        for col in numeric_fields:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # Prétraiter les données
        df_processed = preprocess_data(df, is_train=False)

        # Supprimer Loan_ID si présent
        if "Loan_ID" in df_processed:
            loan_id = df_processed["Loan_ID"].values[0]
            df_processed = df_processed.drop("Loan_ID", axis=1)
        else:
            loan_id = form_data.get("Loan_ID", "N/A")

        # Ajouter les colonnes manquantes
        if trained_columns is None:
            trained_columns = getattr(model, "feature_names_in_", None)
        
        for col in trained_columns:
            if col not in df_processed.columns:
                df_processed[col] = 0

        # Réordonner les colonnes
        df_processed = df_processed[trained_columns]

        # Prédiction
        pred = predict(model, df_processed)[0]
        result = "Approved" if pred == 1 else "Rejected"

        return render_template("result.html", result=result, loan_id=loan_id)

    return render_template("index.html")


# ----------------------------
# Page de Réentraînement
# ----------------------------
@app.route("/retrain", methods=["GET", "POST"])
def retrain():
    global model, trained_columns

    if request.method == "POST":
        # Récupérer les hyperparamètres
        try:
            C = float(request.form.get("C", 1.0))
            max_iter = int(request.form.get("max_iter", 1000))
        except ValueError:
            return "Erreur: Les hyperparamètres doivent être des nombres valides", 400

        # Valider les paramètres
        if C <= 0:
            return "Erreur: C doit être supérieur à 0", 400
        if max_iter <= 0:
            return "Erreur: max_iter doit être supérieur à 0", 400

        # Charger les données
        data = prepare_data("data/train.csv", "data/test.csv")
        X_train, X_test = data["X_train"], data["X_test"]
        y_train, y_test = data["y_train"], data["y_test"]

        # Évaluer l'ancien modèle
        old_results = evaluate_model(model, X_test, y_test)
        old_acc = old_results["accuracy"]
        old_pred = old_results["predictions"]
        old_cm = confusion_matrix(y_test, old_pred).tolist()

        # Entraîner le nouveau modèle
        new_model = retrain_model(X_train, y_train, C=C, max_iter=max_iter)

        # Évaluer le nouveau modèle
        new_results = evaluate_model(new_model, X_test, y_test)
        new_acc = new_results["accuracy"]
        new_pred = new_results["predictions"]
        new_cm = confusion_matrix(y_test, new_pred).tolist()

        # Comparer et garder le meilleur
        if new_acc >= old_acc:
            save_model(
                new_model, 
                file_path="model",
                keep_timestamp=True,
                metrics={"accuracy": new_acc}
            )
            model = load_model()
            trained_columns = getattr(model, "feature_names_in_", None)
            chosen = "new"
            print(f"✅ Nouveau modèle accepté. Accuracy: {new_acc:.4f}")
        else:
            save_model(
                new_model, 
                file_path="model_rejected",
                keep_timestamp=True
            )
            chosen = "old"
            print(f"⚠️ Nouveau modèle rejeté. Accuracy: {new_acc:.4f} < {old_acc:.4f}")

        comparison = {
            "old_model": {
                "accuracy": old_acc, 
                "confusion_matrix": old_cm
            },
            "new_model": {
                "accuracy": new_acc, 
                "confusion_matrix": new_cm, 
                "hyperparameters": {
                    "C": C, 
                    "max_iter": max_iter
                }
            },
            "better_model": chosen
        }

        return render_template("retrain.html", comparison=comparison)

    return render_template("retrain.html", comparison=None)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
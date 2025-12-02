from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
import pandas as pd
from model_pipeline import (
    prepare_data,
    retrain_model,
    evaluate_model,
    save_model,
    load_model,
    preprocess_data,
    predict,
    _read_metadata,
)

import logging
import json
from datetime import datetime
import threading
import os

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, "app.log")),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


MODEL_PATH = "logistic_regression_loan_model.pkl"

app = FastAPI()

model_lock = threading.Lock()

# Charger le modèle au démarrage (si présent)
try:
    model = load_model(MODEL_PATH)
    trained_columns = getattr(model, "feature_names_in_", None)
except FileNotFoundError:
    model = None
    trained_columns = None


class LoanData(BaseModel):
    Loan_ID: str
    Gender: str
    Married: str
    Dependents: str
    Education: str
    Self_Employed: str
    ApplicantIncome: float
    CoapplicantIncome: float
    LoanAmount: float
    Loan_Amount_Term: float | None = None
    Credit_History: float
    Property_Area: str


"""
class RetrainParams(BaseModel):
    C: float = 1.0
    max_iter: int = 100
"""


class RetrainParams(BaseModel):
    C: float = Field(
        1.0, gt=0, description="Inverse of regularization strength (must be > 0)"
    )
    max_iter: int = Field(100, gt=0, description="Max iterations (must be > 0)")


@app.post("/predict")
def predict_loan(data: LoanData):
    global trained_columns

    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        df = pd.DataFrame([data.dict()])

        # Apply training preprocessing (will use preprocessing_params saved by load_model)
        df_processed = preprocess_data(df, is_train=False)

        # Drop Loan_ID
        if "Loan_ID" in df_processed.columns:
            df_processed = df_processed.drop("Loan_ID", axis=1)

        # Ensure all trained columns exist
        if trained_columns is None:
            # try to infer from model if possible
            trained_columns = getattr(model, "feature_names_in_", None)
        for col in trained_columns:
            if col not in df_processed.columns:
                df_processed[col] = 0

        # Reorder
        df_processed = df_processed[trained_columns]

        pred = predict(model, df_processed)[0]
        result = "Approved" if pred == 1 else "Rejected"

        return {"Loan_ID": data.Loan_ID, "prediction": result}

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/retrain")
def retrain_endpoint(params: RetrainParams, background_tasks: BackgroundTasks):
    """
    Lance le réentraînement en tâche de fond.
    """

    # Définir la fonction de tâche en arrière-plan
    def retrain_task(p: RetrainParams):
        global model, trained_columns
        try:
            logger.info(f"Starting retrain with {p.dict()}")
            data = prepare_data("data/train.csv", "data/test.csv")
            X_train, y_train = data["X_train"], data["y_train"]
            X_test, y_test = data["X_test"], data["y_test"]

            # Entraînement
            new_model = retrain_model(X_train, y_train, C=p.C, max_iter=p.max_iter)

            # Évaluation
            results = evaluate_model(new_model, X_test, y_test)
            acc = results.get("accuracy", None)

            # Sauvegarde temporaire du nouveau modèle (sans mettre à jour le "best" automatiquement)
            candidate_path = save_model(
                new_model, file_path=f"candidate_model", keep_timestamp=True
            )

            # Décider si on accepte le nouveau modèle
            meta = _read_metadata()  # from model_pipeline or replicate read logic here
            best_acc = meta.get("best_accuracy", -1)

            if acc is None:
                logger.error("Pas de métrique accuracy retournée — abortion")
                return

            # Si nouveau modèle >= meilleur -> on le marque best
            if acc >= best_acc:
                # update metadata via save_model by passing metrics
                save_model(
                    new_model,
                    file_path=f"model",
                    keep_timestamp=True,
                    metrics={"accuracy": acc},
                )
                # mettre à jour modèle en mémoire avec lock
                with model_lock:
                    model = new_model
                    trained_columns = getattr(model, "feature_names_in_", None)
                logger.info(f"✅ Retrain accepted. New best accuracy: {acc:.4f}")
            else:
                # rollback : garder l'ancien "best", ne pas charger le candidat
                logger.info(
                    f"⚠ Retrain rejected (accuracy {acc:.4f} < best {best_acc}). Candidate saved at {candidate_path}"
                )
            # optionnel : supprimer candidate si tu veux
        except Exception as e:
            logger.exception(f"Erreur lors du réentraînement: {e}")

    # Lancer la tâche en arrière-plan
    background_tasks.add_task(retrain_task, params)

    # Retour HTTP immédiat
    return {
        "status": "retraining_started",
        "hyperparameters": params.dict(),
        "note": "Retraining runs in background. Check logs for result.",
    }

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
import warnings
import pickle
import os

import logging
import json
from datetime import datetime
import threading

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)

MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)
METADATA_PATH = os.path.join(MODELS_DIR, "metadata.json")


def _write_metadata(meta: dict):
    with open(METADATA_PATH, "w") as f:
        json.dump(meta, f, indent=2)


def _read_metadata():
    if not os.path.exists(METADATA_PATH):
        return {}
    with open(METADATA_PATH, "r") as f:
        return json.load(f)


warnings.filterwarnings("ignore")

# Préprocessing

target_encoder = {"Y": 1, "N": 0}
preprocessing_params = {}


def encode_categorical_features(df, is_train=True):
    """
    Encoder les variables catégorielles du DataFrame.

    Args:
        df (pd.DataFrame): DataFrame à encoder
        is_train (bool): True si c'est le jeu d'entraînement, False sinon

    Returns:
        pd.DataFrame: DataFrame avec variables catégorielles encodées
    """
    df_encoded = df.copy()

    # Gender
    if "Gender" in df_encoded.columns:
        mode_val = df_encoded["Gender"].mode()[0] if is_train else "Male"
        df_encoded["Gender"] = df_encoded["Gender"].fillna(mode_val)
        df_encoded["Male"] = (df_encoded["Gender"] == "Male").astype(int)
        df_encoded.drop("Gender", axis=1, inplace=True)

    # Married
    if "Married" in df_encoded.columns:
        mode_val = df_encoded["Married"].mode()[0] if is_train else "Yes"
        df_encoded["Married"] = df_encoded["Married"].fillna(mode_val)
        df_encoded["Married_Yes"] = (df_encoded["Married"] == "Yes").astype(int)
        df_encoded.drop("Married", axis=1, inplace=True)

    # Dependents
    if "Dependents" in df_encoded.columns:
        df_encoded["Dependents"] = df_encoded["Dependents"].fillna("0")
        rpl = {"0": 0, "1": 1, "2": 2, "3+": 3}
        df_encoded["Dependents"] = df_encoded["Dependents"].replace(rpl).astype(int)

    # Education
    if "Education" in df_encoded.columns:
        df_encoded["Education_Graduate"] = (
            df_encoded["Education"] == "Graduate"
        ).astype(int)
        df_encoded.drop("Education", axis=1, inplace=True)

    # Self_Employed
    if "Self_Employed" in df_encoded.columns:
        mode_val = df_encoded["Self_Employed"].mode()[0] if is_train else "No"
        df_encoded["Self_Employed"] = df_encoded["Self_Employed"].fillna(mode_val)
        df_encoded["Self_Employed_Yes"] = (df_encoded["Self_Employed"] == "Yes").astype(
            int
        )
        df_encoded.drop("Self_Employed", axis=1, inplace=True)

    # Property_Area
    if "Property_Area" in df_encoded.columns:
        property_area_dummies = pd.get_dummies(
            df_encoded["Property_Area"], prefix="Property_Area"
        )
        df_encoded = pd.concat([df_encoded, property_area_dummies], axis=1)
        df_encoded.drop("Property_Area", axis=1, inplace=True)

    # Credit_History
    if "Credit_History" in df_encoded.columns:
        mode_val = df_encoded["Credit_History"].mode()[0] if is_train else 1.0
        df_encoded["Credit_History"] = df_encoded["Credit_History"].fillna(mode_val)

    # LoanAmount et Loan_Amount_Term
    if "LoanAmount" in df_encoded.columns:
        mean_val = (
            df_encoded["LoanAmount"].mean()
            if is_train
            else df_encoded["LoanAmount"].mean()
        )
        df_encoded["LoanAmount"] = df_encoded["LoanAmount"].fillna(mean_val)

    if "Loan_Amount_Term" in df_encoded.columns:
        df_encoded.drop("Loan_Amount_Term", axis=1, inplace=True)

    return df_encoded


def handle_missing_values(df, is_train=True):
    """
    Gérer les valeurs manquantes et stocker les paramètres de preprocessing.

    Args:
        df (pd.DataFrame): DataFrame à nettoyer
        is_train (bool): True si c'est le jeu d'entraînement, False sinon

    Returns:
        pd.DataFrame: DataFrame nettoyé
    """
    df_clean = df.copy()
    if is_train:
        preprocessing_params["mean_LoanAmount"] = df_clean["LoanAmount"].mean()
        preprocessing_params["mode_Credit_History"] = df_clean["Credit_History"].mode()[
            0
        ]
    return df_clean


def preprocess_data(df, is_train=True):
    """
    Prétraiter le DataFrame en encodant les variables catégorielles et en gérant les valeurs manquantes.

    Args:
        df (pd.DataFrame): DataFrame à prétraiter
        is_train (bool): True si c'est le jeu d'entraînement, False sinon

    Returns:
        pd.DataFrame: DataFrame prétraité
    """
    df_processed = encode_categorical_features(df, is_train)
    df_processed = handle_missing_values(df_processed, is_train)
    return df_processed


def prepare_data(train_path, test_path):
    """
    Charger et préparer les jeux de données pour l'entraînement et le test.

    Args:
        train_path (str): Chemin vers le fichier CSV d'entraînement
        test_path (str): Chemin vers le fichier CSV de test

    Returns:
        dict: Contient X_train, X_test, y_train, y_test, X_final_test et test_ids
    """
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    logger.info(f"Train shape: {train.shape}, Test shape: {test.shape}")

    train_processed = preprocess_data(train, is_train=True)
    test_processed = preprocess_data(test, is_train=False)

    X = train_processed.drop(["Loan_ID", "Loan_Status"], axis=1, errors="ignore")
    y = train_processed["Loan_Status"].map(target_encoder)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    X_final_test = test_processed.drop(["Loan_ID"], axis=1, errors="ignore")

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "X_final_test": X_final_test,
        "test_ids": (
            test_processed["Loan_ID"] if "Loan_ID" in test_processed.columns else None
        ),
    }


# Modèle


def train_model(X_train, y_train):
    """
    Entraîner un modèle de Régression Logistique sur les données d'entraînement.

    Args:
        X_train (pd.DataFrame): Features d'entraînement
        y_train (pd.Series): Labels d'entraînement

    Returns:
        LogisticRegression: Modèle entraîné
    """
    model = LogisticRegression(random_state=42, max_iter=1000, solver="lbfgs")
    model.fit(X_train, y_train)
    logger.info("Modèle de Régression Logistique entraîné avec succès")
    return model


def evaluate_model(model, X_test, y_test):
    """
    Évaluer les performances du modèle sur les données de test.

    Args:
        model: Modèle entraîné
        X_test (pd.DataFrame): Features de test
        y_test (pd.Series): Labels de test

    Returns:
        dict: Contient l'accuracy, les prédictions et les labels réels
    """
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    logger.info(f"Accuracy du modèle: {accuracy:.4f}")
    logger.info(f"Prédictions correctes: {np.sum(y_test == y_pred)}")
    logger.info(f"Prédictions incorrectes: {np.sum(y_test != y_pred)}")
    return {"accuracy": accuracy, "predictions": y_pred, "actual": y_test}


def predict(model, X_data):
    """
    Faire des prédictions sur de nouvelles données.

    Args:
        model: Modèle entraîné
        X_data (pd.DataFrame): Features pour la prédiction

    Returns:
        np.ndarray: Prédictions du modèle
    """
    return model.predict(X_data)


"""
def save_model(model, file_path):
    
    Sauvegarder le modèle et les paramètres de preprocessing.

    Args:
        model: Modèle entraîné
        file_path (str): Chemin du fichier de sauvegarde
    
    with open(file_path, "wb") as f:
        pickle.dump(
            {
                "model": model,
                "preprocessing_params": preprocessing_params,
                "target_encoder": target_encoder,
            },
            f,
        )
    logger.info(f"Modèle sauvegardé dans: {file_path}")
"""


def save_model(model, file_path=None, keep_timestamp=True, metrics: dict | None = None):
    """
    Sauvegarde le modèle avec versioning. Si keep_timestamp True => models/model_YYYYMMDD_HHMMSS.pkl
    Met à jour metadata.json qui stocke le meilleur model_path et ses metrics.
    """
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    if file_path is None:
        file_path = os.path.join(MODELS_DIR, f"model_{ts}.pkl")
    else:
        if keep_timestamp:
            base = os.path.splitext(file_path)[0]
            file_path = os.path.join(MODELS_DIR, f"{base}_{ts}.pkl")
    with open(file_path, "wb") as f:
        pickle.dump(
            {
                "model": model,
                "preprocessing_params": preprocessing_params,
                "target_encoder": target_encoder,
            },
            f,
        )
    logger.info(f"Modèle sauvegardé dans: {file_path}")

    # mettre à jour metadata (n'ajuste "best" ici QUE si metrics est donné)
    meta = _read_metadata()
    if metrics is not None:
        # si aucune métrique sauvegardée, on prend celui-ci comme best
        best_acc = meta.get("best_accuracy", -1)
        if metrics.get("accuracy", -1) >= best_acc:
            meta.update(
                {
                    "best_model_path": file_path,
                    "best_accuracy": metrics.get("accuracy"),
                    "best_timestamp": ts,
                }
            )
            _write_metadata(meta)
            logger.info(f"Metadata mise à jour : meilleur modèle -> {file_path}")
    return file_path


"""
def load_model(file_path):
    
    Charger un modèle sauvegardé ainsi que les paramètres de preprocessing.

    Args:
        file_path (str): Chemin du fichier sauvegardé

    Returns:
        Modèle chargé
    
    with open(file_path, "rb") as f:
        saved_data = pickle.load(f)
    global preprocessing_params, target_encoder
    preprocessing_params = saved_data["preprocessing_params"]
    target_encoder = saved_data["target_encoder"]
    logger.info(f"Modèle chargé depuis: {file_path}")
    return saved_data["model"]
"""


def load_model(file_path=None):
    """
    Si file_path None -> charge best_model_path depuis metadata, sinon charge explicitement.
    Retourne le model et met à jour preprocessing_params et target_encoder.
    """
    if file_path is None:
        meta = _read_metadata()
        file_path = meta.get("best_model_path")
        if file_path is None:
            raise FileNotFoundError("Aucun modèle trouvé dans metadata.")
    with open(file_path, "rb") as f:
        saved_data = pickle.load(f)
    global preprocessing_params, target_encoder
    preprocessing_params = saved_data.get("preprocessing_params", {})
    target_encoder = saved_data.get("target_encoder", {"Y": 1, "N": 0})
    logger.info(f"Modèle chargé depuis: {file_path}")
    return saved_data["model"]


def create_submission(predictions, test_ids, file_name="submission.csv"):
    """
    Créer un fichier CSV de soumission avec les prédictions.

    Args:
        predictions (array-like): Prédictions du modèle
        test_ids (pd.Series): IDs des échantillons de test
        file_name (str): Nom du fichier de sortie

    Returns:
        pd.DataFrame: DataFrame des prédictions
    """
    submission = pd.DataFrame(
        {
            "Loan_ID": test_ids,
            "Loan_Status": ["Y" if p == 1 else "N" for p in predictions],
        }
    )
    submission.to_csv(file_name, index=False)
    logger.info(f"Fichier de soumission créé : {file_name}")
    return submission


def retrain_model(X_train, y_train, C=1.0, max_iter=100):
    model = LogisticRegression(C=C, max_iter=max_iter)
    model.fit(X_train, y_train)
    return model

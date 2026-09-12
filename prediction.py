from pathlib import Path

import joblib


MODEL_DIR = Path(__file__).resolve().parent.parent / "models"


def load_models():
    """
    Load the three final tuned EBM models.
    """

    models = {
        "financial_health": joblib.load(
            MODEL_DIR / "financial_health_ebm.pkl"
        ),
        "debt_readiness": joblib.load(
            MODEL_DIR / "debt_readiness_ebm.pkl"
        ),
        "emergency_preparedness": joblib.load(
            MODEL_DIR / "emergency_preparedness_ebm.pkl"
        )
    }

    return models


def generate_predictions(models, X):
    """
    Generate predicted classes and probabilities
    for all three financial resilience outcomes.
    """

    results = {}

    for target, model in models.items():

        prediction = model.predict(X)[0]

        probabilities = model.predict_proba(X)[0]

        probability_dict = {
            str(cls): float(prob)
            for cls, prob in zip(model.classes_, probabilities)
        }

        results[target] = {
            "prediction": str(prediction),
            "probabilities": probability_dict
        }

    return results
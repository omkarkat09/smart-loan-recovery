"""Inference helper for Smart Loan Recovery."""

import pandas as pd
import numpy as np
import mlflow
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

MLFLOW_TRACKING_URI = "sqlite:///C:/Users/dell/Documents/GitHub/smart-loan-recovery/mlruns.db"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

def load_model(model_name="loan-default-classifier", stage="Staging"):
    """Load model from MLflow registry."""
    logger.info(f"Loading model '{model_name}' (stage: {stage})...")
    model_uri = f"models:/{model_name}/{stage}"
    try:
        model = mlflow.sklearn.load_model(model_uri)
        return model
    except Exception as e:
        logger.error(f"Failed to load model from registry: {e}")
        raise

def predict(features_df, model=None):
    """
    Predict default probability and assign risk tier / recommended action.
    Returns a DataFrame with the predictions.
    """
    if model is None:
        model = load_model()
        
    y_prob = model.predict_proba(features_df)[:, 1]
    
    results = pd.DataFrame({
        "default_probability": y_prob
    })
    
    # Define thresholds
    conditions = [
        (results["default_probability"] < 0.3),
        (results["default_probability"] >= 0.3) & (results["default_probability"] <= 0.6),
        (results["default_probability"] > 0.6)
    ]
    tiers = ["low", "medium", "high"]
    actions = [
        "Standard Monitoring",
        "Offer Restructuring / Light Touch",
        "Aggressive Collection / Legal Warning"
    ]
    
    results["risk_tier"] = np.select(conditions, tiers, default="unknown")
    results["recommended_action"] = np.select(conditions, actions, default="unknown")
    
    return results

if __name__ == "__main__":
    pass

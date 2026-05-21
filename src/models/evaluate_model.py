"""Module for evaluating loan recovery models and generating artifact plots."""

import os
import logging
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import mlflow
from sklearn.metrics import (
    roc_curve, auc, RocCurveDisplay,
    precision_recall_curve, PrecisionRecallDisplay,
    confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.calibration import CalibrationDisplay
try:
    from fairlearn.metrics import demographic_parity_difference, equalized_odds_difference
except ImportError:
    pass

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def setup_output_dir(output_dir: str) -> Path:
    """Ensure the output directory exists."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    return out_path


def plot_roc_curve(y_true, y_prob, output_dir: str):
    """Plot and save the ROC curve."""
    out_path = setup_output_dir(output_dir)
    
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    
    fig, ax = plt.subplots(figsize=(7, 6))
    display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name='Model')
    display.plot(ax=ax)
    
    ax.set_title("Receiver Operating Characteristic (ROC)")
    ax.plot([0, 1], [0, 1], linestyle='--', color='r', label='Chance')
    ax.legend()
    
    save_path = out_path / "roc_curve.png"
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()
    
    return str(save_path)


def plot_precision_recall_curve(y_true, y_prob, output_dir: str):
    """Plot and save the Precision-Recall curve."""
    out_path = setup_output_dir(output_dir)
    
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    
    fig, ax = plt.subplots(figsize=(7, 6))
    display = PrecisionRecallDisplay(precision=precision, recall=recall, estimator_name='Model')
    display.plot(ax=ax)
    
    ax.set_title("Precision-Recall Curve")
    
    save_path = out_path / "pr_curve.png"
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()
    
    return str(save_path)


def plot_calibration_curve(y_true, y_prob, output_dir: str):
    """Plot and save the calibration curve (reliability diagram)."""
    out_path = setup_output_dir(output_dir)
    
    fig, ax = plt.subplots(figsize=(7, 6))
    CalibrationDisplay.from_predictions(y_true, y_prob, n_bins=10, ax=ax, name='Model')
    
    ax.set_title("Calibration Curve (Reliability Diagram)")
    
    save_path = out_path / "calibration_curve.png"
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()
    
    return str(save_path)


def plot_confusion_matrix_at_thresholds(y_true, y_prob, output_dir: str, thresholds=[0.3, 0.5, 0.7]):
    """Plot and save confusion matrices at various decision thresholds."""
    out_path = setup_output_dir(output_dir)
    
    n_thresholds = len(thresholds)
    fig, axes = plt.subplots(1, n_thresholds, figsize=(5 * n_thresholds, 4))
    
    if n_thresholds == 1:
        axes = [axes]
        
    saved_paths = []
    for ax, thresh in zip(axes, thresholds):
        y_pred = (y_prob >= thresh).astype(int)
        cm = confusion_matrix(y_true, y_pred)
        display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No Default', 'Default'])
        display.plot(ax=ax, cmap='Blues', values_format='d', colorbar=False)
        ax.set_title(f"Threshold = {thresh}")
    
    plt.tight_layout()
    save_path = out_path / "confusion_matrices.png"
    plt.savefig(save_path, dpi=120)
    plt.close()
    
    return str(save_path)


def generate_evaluation_report(y_true, y_prob, run_name="model_evaluation"):
    """
    Generate all evaluation plots, save them locally to docs/model_evaluation, 
    and log them as artifacts to MLflow.
    """
    logger.info("Generating evaluation report plots...")
    
    output_dir = "C:/Users/dell/Documents/GitHub/smart-loan-recovery/docs/model_evaluation"
    
    roc_path = plot_roc_curve(y_true, y_prob, output_dir)
    pr_path = plot_precision_recall_curve(y_true, y_prob, output_dir)
    calib_path = plot_calibration_curve(y_true, y_prob, output_dir)
    cm_path = plot_confusion_matrix_at_thresholds(y_true, y_prob, output_dir, thresholds=[0.3, 0.5, 0.7])
    
    plots = [roc_path, pr_path, calib_path, cm_path]
    
    logger.info(f"Saved {len(plots)} plots to {output_dir}")
    
    # Log to MLflow if an active run exists, otherwise start a new one
    active_run = mlflow.active_run()
    if active_run:
        for plot in plots:
            mlflow.log_artifact(plot, artifact_path="evaluation_plots")
        logger.info("Logged plots to active MLflow run.")
    else:
        with mlflow.start_run(run_name=run_name):
            for plot in plots:
                mlflow.log_artifact(plot, artifact_path="evaluation_plots")
        logger.info(f"Started new MLflow run '{run_name}' and logged plots.")

    return plots


def check_bias(y_true, y_pred, sensitive_features_df):
    """
    Check demographic_parity_difference and equalized_odds_difference on sensitive features.
    If dp_diff > 0.1, print a warning and suggest which features to remove.
    """
    logger.info("Running Fairlearn bias detection...")
    issues_found = False
    for col in sensitive_features_df.columns:
        try:
            dp_diff = demographic_parity_difference(y_true, y_pred, sensitive_features=sensitive_features_df[col])
            eo_diff = equalized_odds_difference(y_true, y_pred, sensitive_features=sensitive_features_df[col])
            
            logger.info(f"Bias Check [{col}]: DP Diff = {dp_diff:.4f} | EO Diff = {eo_diff:.4f}")
            
            if dp_diff > 0.1:
                logger.warning(f"!!! BIAS DETECTED !!! Demographic Parity Difference for '{col}' is {dp_diff:.4f} (> 0.1).")
                logger.warning(f"Recommendation: Consider removing '{col}' from the training features or apply bias mitigation techniques (e.g., Fairlearn Reductions) before accepting this model.")
                issues_found = True
        except Exception as e:
            logger.error(f"Error computing bias for {col}: {e}")
            
    if not issues_found:
        logger.info("Bias checks passed. No significant disparities detected.")
    
    return issues_found


import pandas as pd
import numpy as np
import shap
from pathlib import Path
import mlflow
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from typing import List, Union, Dict, Any
from scipy.stats import loguniform, randint, uniform
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, log_loss, confusion_matrix
)
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score
)
import tempfile

def evaluate_model(
    model: XGBClassifier,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    skip_features: List[str],
    cost_matrix: dict
) -> dict:
    """Evaluate model"""
    # Convert to DataFrame if not already
    if not isinstance(X_test, pd.DataFrame):
        X_test = pd.DataFrame(X_test)
    
    # Filter features (same as training)
    features_to_keep = [col for col in X_test.columns if col not in skip_features]
    X_test_filtered = X_test[features_to_keep]
    
    # Generate predictions
    y_pred = model.predict(X_test_filtered)
    y_proba = model.predict_proba(X_test_filtered)[:, 1]

    # Calculate confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    cm_counts = {
        "true_negative": int(cm[0, 0]),
        "false_positive": int(cm[0, 1]),
        "false_negative": int(cm[1, 0]),
        "true_positive": int(cm[1, 1])
    }
    
    # Calculate cost
    total_cost = calculate_model_cost(cm_counts, cost_matrix)
    cost_per_instance = total_cost / len(y_test)
    
    # Calculate metrics
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "log_loss": log_loss(y_test, y_proba),
        "model_type": "xgboost"
    }

    # Create and log confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Log metrics to MLflow
    for name, value in metrics.items():
        if name != "model_type":
            mlflow.log_metric(name, value)

    # Add confusion matrix values to metrics
    metrics.update({
        "true_negative": int(cm[0, 0]),
        "false_positive": int(cm[0, 1]),
        "false_negative": int(cm[1, 0]),
        "true_positive": int(cm[1, 1])
    })
    
    # Log to MLflow
    mlflow.log_metrics({
        "total_cost": total_cost,
        "cost_per_instance": cost_per_instance
    })
    mlflow.log_dict(cost_matrix, "cost_matrix.json")
    log_confusion_matrix(y_test, y_pred, "confusion_matrix")

    metrics_shap = evaluate_model_with_shap(
        model, X_test, y_test, skip_features, cost_matrix
    )
    metrics.update(metrics_shap)
    return metrics

def evaluate_regression_model(model, X_test, y_test, skip_features):
    """Evaluate model performance and log results"""

    features_to_keep = [col for col in X_test.columns if col not in skip_features]
    X_test_filtered = X_test[features_to_keep]
    y_pred = model.predict(X_test_filtered)
    
    metrics = {
        "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
        "mae": mean_absolute_error(y_test, y_pred),
        "r2": r2_score(y_test, y_pred),
        "mean_actual": np.mean(y_test),
        "mean_predicted": np.mean(y_pred)
    }
    
    # Log metrics
    mlflow.log_metrics(metrics)
    
    # Residual plot
    residuals = y_test - y_pred
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y_pred, y=residuals)
    plt.axhline(0, color='r', linestyle='--')
    plt.xlabel("Predicted Values")
    plt.ylabel("Residuals")
    plt.title("Residual Analysis")
    mlflow.log_figure(plt.gcf(), "residual_plot.png")
    plt.close()
    
    # SHAP analysis (sample for efficiency)
    features_to_keep = [col for col in X_test.columns 
                       if col not in skip_features and
                       X_test[col].dtype.kind in 'bifc']  # Only bool, int, float, complex
    
    X_test_numeric = X_test[features_to_keep].astype(float)

    sample_idx = np.random.choice(len(X_test_numeric), min(500, len(X_test_numeric)), replace=False)
    explainer = shap.Explainer(model, X_test_numeric.iloc[sample_idx])
    shap_values = explainer(X_test_numeric.iloc[sample_idx])
    
    plt.figure()
    shap.summary_plot(shap_values, X_test_numeric.iloc[sample_idx], show=False)
    mlflow.log_figure(plt.gcf(), "shap_summary.png")
    plt.close()
    
    return metrics


def evaluate_model_with_shap(
    model, 
    X_test: pd.DataFrame, 
    y_test: pd.Series,
    skip_features: List[str],
    cost_matrix: dict,
    n_samples: int = 1000  # Use subset for faster computation
) -> dict:
    """
    Evaluate model with SHAP analysis and cost metrics
    """

    # 1. Filter and prepare data
    features_to_keep = [col for col in X_test.columns 
                       if col not in skip_features and
                       X_test[col].dtype.kind in 'bifc']  # Only bool, int, float, complex
    
    X_test_numeric = X_test[features_to_keep].astype(float)
    
    
    # Sample data for faster SHAP computation
    if len(X_test_numeric) > n_samples:
        sample_idx = np.random.choice(len(X_test_numeric), n_samples, replace=False)
        X_shap = X_test_numeric.iloc[sample_idx]
    else:
        X_shap = X_test_numeric
    
    # Generate SHAP values
    explainer = shap.Explainer(model, X_shap)
    shap_values = explainer(X_shap)
    
    # Create output directory
    shap_dir = Path("data/08_reporting/shap_plots")
    shap_dir.mkdir(exist_ok=True)
    
    # 1. Summary Plot
    plt.figure()
    shap.summary_plot(shap_values, X_shap, show=False)
    summary_path = shap_dir / "shap_summary.png"
    plt.tight_layout()
    plt.savefig(summary_path)
    plt.close()
    
    # 2. Feature Importance Plot
    plt.figure()
    shap.plots.bar(shap_values, show=False)
    bar_path = shap_dir / "shap_feature_importance.png"
    plt.tight_layout()
    plt.savefig(bar_path)
    plt.close()
    
    # Log to MLflow
    mlflow.log_artifact(str(summary_path))
    mlflow.log_artifact(str(bar_path))

    # Add SHAP metadata
    metrics = {
        "shap_mean_abs": {feature: np.mean(np.abs(shap_values.values[:, i]))
                          for i, feature in enumerate(X_shap.columns)},
        "shap_n_samples": len(X_shap)
    }
    
    return metrics


def log_feature_importance(model, feature_names):
    importance = model.coef_[0]
    plt.barh(feature_names, importance)
    plt.title("Feature Importance")
    plt.tight_layout()
    
    # Save and log the plot
    plt.savefig("feature_importance.png")
    mlflow.log_artifact("feature_importance.png")
    plt.close()



def log_confusion_matrix(y_test, y_pred, name):
    # Create and log confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Predicted 0', 'Predicted 1'],
                yticklabels=['Actual 0', 'Actual 1'])
    plt.title('Confusion Matrix')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
    # Save to temporary file and log to MLflow
    with tempfile.NamedTemporaryFile(suffix='.png') as tmpfile:
        plt.savefig(tmpfile.name, bbox_inches='tight', dpi=300)
        mlflow.log_artifact(tmpfile.name, name)
    plt.close()


def calculate_model_cost(
    confusion_matrix: Dict[str, int],
    cost_matrix: Dict[str, float]
) -> float:
    """
    Calculate total cost based on confusion matrix counts and cost matrix
    
    Args:
        confusion_matrix: Dictionary with keys:
            - true_positive
            - true_negative
            - false_positive
            - false_negative
        cost_matrix: Dictionary with cost/profit values for each outcome
    
    Returns:
        Total cost of the model
    """
    return (
        (confusion_matrix["true_positive"] * cost_matrix["true_positive"]) +
        (confusion_matrix["true_negative"] * cost_matrix["true_negative"]) +
        (confusion_matrix["false_positive"] * cost_matrix["false_positive"]) +
        (confusion_matrix["false_negative"] * cost_matrix["false_negative"])
    )

def tune_threshold(model, X_val, y_val, skip_features, cost_matrix):
    """Find optimal threshold based on validation set"""
    
    # Filter features (same as training)
    features_to_keep = [col for col in X_val.columns if col not in skip_features]
    X_test_filtered = X_val[features_to_keep]

    # Get predicted probabilities
    y_proba = model.predict_proba(X_test_filtered)[:, 1]
    
    # Test thresholds from 0.01 to 0.99
    thresholds = np.linspace(0.01, 0.99, 100)
    costs = []

    
    for thresh in thresholds:
        y_pred = (y_proba >= thresh).astype(int)
        # Calculate confusion matrix
        cm = confusion_matrix(y_val, y_pred)
        cm_counts = {
            "true_negative": int(cm[0, 0]),
            "false_positive": int(cm[0, 1]),
            "false_negative": int(cm[1, 0]),
            "true_positive": int(cm[1, 1])
        }
        costs.append(calculate_model_cost(cm_counts, cost_matrix))
    
    # Find optimal threshold
    optimal_idx = np.argmax(costs)
    optimal_threshold = thresholds[optimal_idx]
    min_cost = costs[optimal_idx]
    min_cost_per_instance = min_cost / len(y_val)
    
    # Create plot
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, costs, 'b-')
    plt.axvline(optimal_threshold, color='r', linestyle='--', 
                label=f'Optimal Threshold: {optimal_threshold:.2f}')
    plt.xlabel('Threshold')
    plt.ylabel('Cost per Instance')
    plt.title('Threshold Tuning Based on Cost')
    plt.legend()
    plt.grid(True)
    
    # Log threshold plot
    mlflow.log_figure(plt.gcf(), "threshold_tuning.png")
    mlflow.log_param("optimal_threshold", optimal_threshold)
    mlflow.log_metric("min_cost", min_cost)
    mlflow.log_metric("min_cost_per_instance", min_cost_per_instance)

    return optimal_threshold, plt.gcf()


def evaluate_with_optimal_threshold(model, X_test, y_test, threshold, skip_features):
    """Evaluate model with custom threshold"""

    # Filter features (same as training)
    features_to_keep = [col for col in X_test.columns if col not in skip_features]
    X_test_filtered = X_test[features_to_keep]

    y_proba = model.predict_proba(X_test_filtered)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)
    
    log_confusion_matrix(y_test, y_pred, "confusion_matrix_tuned")
    # Calculate metrics
    metrics = {
        "tuned_accuracy": accuracy_score(y_test, y_pred),
        "tuned_precision": precision_score(y_test, y_pred),
        "tuned_recall": recall_score(y_test, y_pred),
        "tuned_f1": f1_score(y_test, y_pred),
        "tuned_roc_auc": roc_auc_score(y_test, y_proba),
        "tuned_log_loss": log_loss(y_test, y_proba),
    }
    
    # Log metrics to MLflow
    for name, value in metrics.items():
        if name != "model_type":
            mlflow.log_metric(name, value)
    
    return metrics


import pandas as pd
import numpy as np
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

    log_confusion_matrix(y_test, y_pred)
    
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



def log_confusion_matrix(y_test, y_pred):
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
        mlflow.log_artifact(tmpfile.name, "confusion_matrix")
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

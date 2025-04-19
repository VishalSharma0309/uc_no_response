import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from typing import List, Union, Dict, Any
from scipy.stats import loguniform, randint, uniform
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, log_loss, confusion_matrix
)
import mlflow
import json
import tempfile

def train_logistic_regression_model(
    X_train: Union[pd.DataFrame, np.ndarray],
    y_train: Union[pd.Series, np.ndarray],
    parameters: dict,
    skip_features: List[str]
) -> LogisticRegression:
    """
    Train logistic regression model after removing specified features
    
    Args:
        X_train: Training features
        y_train: Training labels
        parameters: Model hyperparameters
        skip_features: List of features to exclude
        
    Returns:
        Trained LogisticRegression model
    """
    # Convert to DataFrame if not already (handles case where X_train is numpy array)
    if not isinstance(X_train, pd.DataFrame):
        X_train = pd.DataFrame(X_train)
    
    # Remove skip features
    features_to_keep = [col for col in X_train.columns if col not in skip_features]
    X_train_filtered = X_train[features_to_keep]
    
    # Log feature information
    mlflow.log_param("n_features_original", X_train.shape[1])
    mlflow.log_param("n_features_after_selection", len(features_to_keep))
    mlflow.log_param("features_removed", skip_features)
    mlflow.log_param("features_used", features_to_keep)
    
    # Initialize and train model
    model = LogisticRegression(**parameters)
    model.fit(X_train_filtered, y_train)
    
    mlflow.sklearn.log_model(model, "logistic_regression_model")
    log_feature_importance(model, features_to_keep)
    
    return model
    

def evaluate_model(
    model: LogisticRegression,
    X_test: Union[pd.DataFrame, np.ndarray],
    y_test: Union[pd.Series, np.ndarray],
    skip_features: List[str]
) -> dict:
    """
    Evaluate model performance after removing specified features
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
        skip_features: List of features to exclude (must match training)
        
    Returns:
        Dictionary of evaluation metrics
    """
    # Convert to DataFrame if not already
    if not isinstance(X_test, pd.DataFrame):
        X_test = pd.DataFrame(X_test)
    
    # Remove same features as training
    features_to_keep = [col for col in X_test.columns if col not in skip_features]
    X_test_filtered = X_test[features_to_keep]
    
    # Generate predictions and calculate metrics
    y_pred = model.predict(X_test_filtered)
    y_proba = model.predict_proba(X_test_filtered)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "log_loss": log_loss(y_test, y_proba),
        "model_type": "logistic_regression"
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
    
    return metrics


def train_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    parameters: dict,
    skip_features: List[str]
) -> XGBClassifier:
    """Train XGBoost classifier with feature selection"""
    # Filter features
    features_to_keep = [col for col in X_train.columns if col not in skip_features]
    X_train_filtered = X_train[features_to_keep]
    
    # Log feature info
    mlflow.log_param("model_type", "xgboost")
    mlflow.log_params(parameters)
    mlflow.log_param("n_features", len(features_to_keep))
    
    # Train model
    model = XGBClassifier(**parameters)
    print(X_train_filtered.columns)
    model.fit(X_train_filtered, y_train)
    
    # Log feature importance
    importance = model.feature_importances_
    for feature, score in zip(features_to_keep, importance):
        mlflow.log_metric(f"feature_importance_{feature}", score)
    
    mlflow.sklearn.log_model(model, "xgboost_classifier_model")

    return model


def evaluate_xgboost(
    model: XGBClassifier,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    skip_features: List[str],
    cost_matrix: dict
) -> dict:
    """Evaluate XGBoost model"""
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

def tune_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    skip_features: List[str],
    cv_params: Dict[str, Any],
    param_distributions: Dict[str, Any],
    fixed_params: Dict[str, Any]
) -> Dict:
    """
    Perform grid search with cross-validation for XGBoost
    
    Args:
        X_train: Training features
        y_train: Training labels
        skip_features: Features to exclude
        cv_params: Cross-validation parameters
        grid_params: Hyperparameter search space
        fixed_params: Fixed model parameters
        
    Returns:
        Dictionary containing:
        - best_model: Tuned XGBClassifier
        - best_params: Best parameters found
        - cv_results: Full CV results
    """
    # Filter features
    features_to_keep = [col for col in X_train.columns if col not in skip_features]
    X_train_filtered = X_train[features_to_keep]
    
    # Convert parameter config to scipy distributions
    param_dist = get_param_distributions(param_distributions)
    

    # Initialize base model
    model = XGBClassifier(**fixed_params)
    
    # Setup random search
    random_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_dist,
        n_iter=cv_params["n_iter"],
        cv=cv_params["n_folds"],
        scoring=cv_params["scoring"],
        n_jobs=cv_params["n_jobs"],
        random_state=cv_params["random_state"],
        verbose=1,
        return_train_score=True
    )
    
    # Perform search
    random_search.fit(X_train_filtered, y_train)
    
    # MLflow logging
    with mlflow.start_run(nested=True):
        # Log search configuration
        mlflow.log_params({
            "search_strategy": "randomized",
            "n_iter": cv_params["n_iter"],
            "cv_folds": cv_params["n_folds"]
        })
        
        # Log best parameters and metrics
        mlflow.log_params({f"best_{k}": v for k, v in random_search.best_params_.items()})
        mlflow.log_metric("best_cv_score", random_search.best_score_)
        
        # Log parameter distributions
        mlflow.log_dict(param_distributions, "param_distributions.json")
        
        # Process cv_results
        cv_results = pd.DataFrame(random_search.cv_results_)
        
        # Split into metrics and params
        metrics_cols = [col for col in cv_results.columns if not col.startswith('param_')]
        params_cols = [col for col in cv_results.columns if col.startswith('param_')]
            
        # Create and log final model
        final_model = XGBClassifier(**fixed_params)
        final_model.set_params(**random_search.best_params_)
        final_model.fit(X_train_filtered, y_train)
        
    mlflow.xgboost.log_model(final_model, "xgboost_tuned_model")   
    
    return {
        "best_model": final_model,
        "best_params": random_search.best_params_,
        "cv_results": random_search.cv_results_
    }


def get_param_distributions(param_config: Dict) -> Dict:
    """Convert parameter configuration to scipy distributions"""
    param_dist = {}
    for param, config in param_config.items():
        if config["type"] == "loguniform":
            param_dist[param] = loguniform(config["values"][0], config["values"][1])
        elif config["type"] == "randint":
            param_dist[param] = randint(config["values"][0], config["values"][1])
        elif config["type"] == "uniform":
            param_dist[param] = uniform(config["values"][0], 
                                      config["values"][1] - config["values"][0])
    return param_dist

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

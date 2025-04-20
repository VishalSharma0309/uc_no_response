
import pandas as pd
import numpy as np
import mlflow
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from typing import List, Union, Dict, Any
from scipy.stats import loguniform, randint, uniform

from ..evaluation.nodes import log_feature_importance

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


def train_xgboost_regressor(X_train, y_train, X_test, y_test, params, skip_features):
    """Train XGBoost regression model"""
    features_to_keep = [col for col in X_test.columns if col not in skip_features]
    X_filtered = X_train[features_to_keep]
    X_test_filtered = X_test[features_to_keep]
    
    model = XGBRegressor(**params)
    model.fit(
        X_filtered, y_train,
        eval_set=[(X_test_filtered, y_test)],
        verbose=10
    )
    
    # Log feature importance
    fig, ax = plt.subplots(figsize=(10, 8))
    xgb.plot_importance(model, ax=ax)
    mlflow.log_figure(fig, "feature_importance.png")
    plt.close()

    mlflow.xgboost.log_model(model, "xgboost_regressor")
    
    return model


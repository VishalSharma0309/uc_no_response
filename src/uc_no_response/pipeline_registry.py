"""Project pipelines."""
from __future__ import annotations

from kedro.framework.project import find_pipelines
from kedro.pipeline import Pipeline

from uc_no_response.pipelines import (
    logistic_regression_data_processing_pipeline,
    xgboost_classifier_data_processing_pipeline,
    xgboost_regressor_data_processing_pipeline,
    logistic_regression_training_pipeline,
    xgboost_classifier_training_pipeline,
    xgboost_regressor_training_pipeline
)

def register_pipelines() -> dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """
    pipelines = find_pipelines()
    pipelines["logistic_regression_data_processing"] = logistic_regression_data_processing_pipeline()
    pipelines["xgboost_classifier_data_processing"] = xgboost_classifier_data_processing_pipeline()
    pipelines["xgboost_regressor_data_processing_pipeline"] = xgboost_regressor_data_processing_pipeline()
    pipelines["logistic_regression_training"] = logistic_regression_training_pipeline()
    pipelines["xgboost_classifier_training"] = xgboost_classifier_training_pipeline()
    pipelines["xgboost_regressor_training_pipeline"] = xgboost_regressor_training_pipeline()

    pipelines["train_logistic_regression"] = pipelines["logistic_regression_data_processing"] + pipelines["logistic_regression_training"]
    pipelines["train_xgboost_classifier"] = pipelines["xgboost_classifier_data_processing"] + pipelines["xgboost_classifier_training"]
    pipelines["train_xgboost_regression"] = pipelines["xgboost_regressor_data_processing_pipeline"] + pipelines["xgboost_regressor_training_pipeline"]
    return pipelines

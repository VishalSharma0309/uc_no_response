from kedro.pipeline import Pipeline, node, pipeline
from .nodes import (
    train_logistic_regression_model,
    evaluate_model,
    train_xgboost, 
    tune_xgboost,
    evaluate_xgboost,
)

def xgboost_classifier_training_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            # node(
            #     func=train_xgboost,
            #     inputs=["X_train", "y_train", "params:model_params.xgboost", "params:data_preprocessing.skip_features"],
            #     outputs="xgboost_classifier",
            #     name="train_xgboost_node"
            # ),
            node(
                func=tune_xgboost,
                inputs=[
                    "X_train",
                    "y_train",
                    "params:data_preprocessing.skip_features",
                    "params:model_params.xgboost.cv_params",
                    "params:model_params.xgboost.param_distributions",
                    "params:model_params.xgboost.fixed_params"
                ],
                outputs={
                    "best_model": "xgboost_tuned",
                    "best_params": "xgboost_best_params",
                    "cv_results": "xgboost_cv_results"
                },
                name="tune_xgboost_node"
            ),
            node(
                func=evaluate_xgboost,
                inputs=["xgboost_tuned", 
                        "X_test", "y_test", 
                        "params:data_preprocessing.skip_features",
                        "params:cost_matrix"],
                outputs="xgboost_metrics",
                name="evaluate_xgboost_node"
            )
        ]
    )

def logistic_regression_training_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=train_logistic_regression_model,
                inputs=["X_train", "y_train", "params:model_params.logistic_regression", "params:data_preprocessing.skip_features"],
                outputs="classifier",
                name="train_logistic_regression_node",
            ),
            node(
                func=evaluate_model,
                inputs=["classifier", "X_test", "y_test", "params:data_preprocessing.skip_features"],
                outputs="metrics",
                name="evaluate_logistic_regression_node"
            )
        ]
    )
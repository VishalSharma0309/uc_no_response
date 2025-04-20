from kedro.pipeline import Pipeline, node, pipeline
from .nodes import (
    train_logistic_regression_model,
    train_xgboost_regressor,
    train_xgboost, 
    tune_xgboost,
)
from ..evaluation.nodes import (
    evaluate_model,
    tune_threshold,
    evaluate_with_optimal_threshold,
    evaluate_regression_model
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
                func=evaluate_model,
                inputs=["xgboost_tuned", 
                        "X_test", "y_test", 
                        "params:data_preprocessing.skip_features",
                        "params:cost_matrix"],
                outputs="xgboost_metrics",
                name="evaluate_xgboost_node"
            ),
            node(
                func=tune_threshold,
                inputs=["xgboost_tuned", "X_test", "y_test", "params:data_preprocessing.skip_features", "params:cost_matrix"],
                outputs=["optimal_threshold", "threshold_plot"],
                name="tune_threshold_node"
            ),
            node(
                func=evaluate_with_optimal_threshold,
                inputs=["xgboost_tuned", "X_test", "y_test", "optimal_threshold", "params:data_preprocessing.skip_features"],
                outputs="threshold_metrics",
                name="evaluate_with_threshold_node"
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
                inputs=["classifier", 
                        "X_test", "y_test", 
                        "params:data_preprocessing.skip_features",
                        "params:cost_matrix"],
                outputs="logistic_regression_metrics",
                name="evaluate_xgboost_node"
            )
        ]
    )


def xgboost_regressor_training_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=train_xgboost_regressor,
                inputs=["X_train", "y_train", "X_test", "y_test",
                        "params:model_params.xgboost", "params:xgboost_regression.feature_selection.skip_features"],
                outputs="xgboost_regressor",
                name="train_xgboost_node"
            ),
            node(
                func=evaluate_regression_model,
                inputs=["xgboost_regressor", 
                        "X_test", "y_test", 
                        "params:xgboost_regression.feature_selection.skip_features"],
                outputs="xgboost_regressor_metrics",
                name="evaluate_xgboost_node"
            )
        ]
    )
from kedro.pipeline import Pipeline, node, pipeline
from .nodes import (
    fill_null_values,
    create_dummy_variables,
    normalize_numerical_columns,
    split_data,
)

def logistic_regression_data_processing_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=fill_null_values,
                inputs=["raw_data", "params:data_preprocessing.user_id_column",
                        "params:data_preprocessing.numerical_features",
                        "params:data_preprocessing.categorical_features"],
                outputs="data_with_filled_nulls",
                name="fill_null_values_node",
            ),
            node(
                func=create_dummy_variables,
                inputs=["data_with_filled_nulls",
                        "params:data_preprocessing.categorical_features"],
                outputs="data_with_dummies",
                name="create_dummy_variables_node",
            ),
            node(
                func=normalize_numerical_columns,
                inputs=["data_with_dummies",
                        "params:data_preprocessing.numerical_features"],
                outputs="preprocessed_data",
                name="normalize_numerical_columns_node",
            ),
            node(
                func=split_data,
                inputs=["preprocessed_data", 
                        "params:target_column", 
                        "params:data_preprocessing.test_size", 
                        "params:model_params.logistic_regression.random_state"],
                outputs=["X_train", "X_test", "y_train", "y_test"],
                name="preprocess_data_node",
            )
        ]
    )

def xgboost_classifier_data_processing_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=create_dummy_variables,
                inputs=["raw_data",
                        "params:data_preprocessing.categorical_features"],
                outputs="data_with_dummies",
                name="create_dummy_variables_node",
            ),
            node(
                func=split_data,
                inputs=["data_with_dummies", 
                        "params:target_column", 
                        "params:data_preprocessing.test_size", 
                        "params:model_params.xgboost.fixed_params.random_state"],
                outputs=["X_train", "X_test", "y_train", "y_test"],
                name="preprocess_data_node",
            )
        ]
    )
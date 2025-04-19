import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Dict, List
from sklearn.preprocessing import MinMaxScaler

def treat_null_values(
    data: pd.DataFrame,
    mapping_features: List[List],
    coexisting_features: List[str],
    numerical_features: List[str],
    categorical_features: List[str]
) -> pd.DataFrame:
    """
    Treat null values by:
    1. First, remove any rows with more than 80% of features as null
    2. Then convert aov to zero if deliveries is zero
    3. Remove coexisting nulls in coexisting features
    
    Args:
        data: Input DataFrame
        coexisting_features: list[str]
    
    Returns:
        pd.DataFrame: DataFrame with null values filled
    """
    # Make a copy to avoid SettingWithCopyWarning
    filled_data = data.copy()
    
    # Step 1: Remove rows with more than 80% null values
    threshold = 0.8 * len(filled_data.columns)
    filled_data = filled_data.dropna(thresh=len(filled_data.columns) - threshold + 1)

    # Step 2: Convert aov to zero if deliveries is zero
    for delivery_col, aov_col in mapping_features:
        filled_data.loc[filled_data[delivery_col] == 0, aov_col] = 0

    # Step 3: Coexisting features with all nulls can be removed
    filled_data.dropna(inplace=True, subset=coexisting_features)

    # Step 4: Fill remaining nulls with median for numerical features
    for column in numerical_features:
        filled_data[column] = filled_data[column].fillna(filled_data[column].median())
    
    for column in categorical_features:
        filled_data[column] = filled_data[column].fillna(filled_data[column].mode()[0])
    
    return filled_data




def create_dummy_variables(
    data: pd.DataFrame,
    categorical_features: List[str]
) -> pd.DataFrame:
    """
    Create dummy variables for categorical features with advanced name sanitization
    
    Args:
        data: Input DataFrame
        categorical_features: List of categorical column names to convert to dummies
        
    Returns:
        pd.DataFrame: DataFrame with properly sanitized dummy variables
    """
    # Make a copy to avoid modifying original DataFrame
    processed_data = data.copy()
    
    # Create dummies for each categorical feature (with original names first)
    for feature in categorical_features:
        if feature in processed_data.columns:
            # Get unique values to determine if dummy creation is needed
            unique_values = processed_data[feature].nunique()
            
            if unique_values > 1:
                # Create dummies with original feature names
                dummies = pd.get_dummies(
                    processed_data[feature],
                    prefix=feature,
                    drop_first=True
                )
                
                # Now sanitize the dummy column names
                dummies.columns = [
                    col.replace('[', '').replace(']', '')
                       .replace('<', '_less_than_')
                       .replace('>', '_more_than_')
                       .replace('+', '_plus')
                    for col in dummies.columns
                ]
                
                processed_data = pd.concat(
                    [processed_data.drop(feature, axis=1), dummies],
                    axis=1
                )
            else:
                # If only one value, drop the column completely
                processed_data = processed_data.drop(feature, axis=1)
                print(f"Warning: Dropped column '{feature}' as it only had one unique value")
    
    # Additional sanitization for any remaining columns
    processed_data.columns = [
        col.replace('[', '').replace(']', '')
           .replace('<', '_less_than_')
           .replace('>', '_more_than_')
           .replace('+', '_plus')
        for col in processed_data.columns
    ]
    
    return processed_data

# def create_dummy_variables(
#     data: pd.DataFrame,
#     categorical_features: List[str]
# ) -> pd.DataFrame:
#     """
#     Create dummy variables for categorical features
    
#     Args:
#         data: Input DataFrame
#         categorical_features: List of categorical column names to convert to dummies
    
#     Returns:
#         pd.DataFrame: DataFrame with dummy variables
#     """
#     # Create dummies for each categorical feature
#     for feature in categorical_features:
#         if feature in data.columns:
#             dummies = pd.get_dummies(data[feature], prefix=feature, drop_first=True)
#             data = pd.concat([data.drop(feature, axis=1), dummies], axis=1)
    
#     return data



def normalize_numerical_columns(
    data: pd.DataFrame,
    columns_to_normalize: List[str]
) -> pd.DataFrame:
    """
    Normalize numerical columns using MinMax scaling
    
    Args:
        data: Input DataFrame
        columns_to_normalize: List of numerical column names to normalize
    
    Returns:
        pd.DataFrame: DataFrame with normalized numerical columns
    """
    # Make a copy to avoid modifying original data
    normalized_data = data.copy()
    
    # Initialize scaler
    scaler = MinMaxScaler()
    
    # Normalize specified columns
    for column in columns_to_normalize:
        if column in normalized_data.columns:
            normalized_data[column] = scaler.fit_transform(normalized_data[[column]])
    
    return normalized_data



def split_data(data: pd.DataFrame, target_column: str, test_size: float = 0.2, random_state: int = 42):
    """Split into train/test sets."""
    X = data.drop(columns=[target_column])
    y = data[target_column]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    return X_train, X_test, y_train, y_test
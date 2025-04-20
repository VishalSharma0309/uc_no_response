import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Dict, List, Tuple
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


def create_engineered_features(data: pd.DataFrame) -> Tuple[pd.DataFrame, list]:
    """
    Creates new features with zero-division protection
    Returns:
        - DataFrame with new features
        - List of new feature names
    """
    df = data.copy()
    new_features = []
    
    # 1. Delivery Intensity
    df['delivery_intensity'] = np.where(
        df['no_of_deliveries_last_365d'] > 0,
        df['no_of_deliveries_last_30d'] / df['no_of_deliveries_last_365d'],
        0  # Default when no yearly deliveries
    )
    new_features.append('delivery_intensity')
    
    # 2. NR (Non-Response) Frequency
    df['NR_frequency'] = np.where(
        df['no_of_deliveries_last_365d'] > 0,
        df['NR_experienced_last_365d'] / df['no_of_deliveries_last_365d'],
        0
    )
    new_features.append('NR_frequency')
    
    # 3. AOV Change (30d vs 365d)
    df['aov_change_30d'] = np.where(
        df['service_deliverd_aov_last_365d'] > 0,
        (df['service_deliverd_aov_last_30d'] - df['service_deliverd_aov_last_365d']) / 
        df['service_deliverd_aov_last_365d'],
        0
    )
    new_features.append('aov_change_30d')
    
    # 4. LTV/AOV Ratio
    df['ltv_aov_ratio'] = np.where(
        df['service_deliverd_aov_last_365d'] > 0,
        df['revenues_next_6_months'] / df['service_deliverd_aov_last_365d'],
        0
    )
    new_features.append('ltv_aov_ratio')
    
    # 5. Customer Tenure Categories
    bins = [0, 500, 1000, 1500, 2000, np.inf]
    labels = ['<500', '500-1000', '1000-1500', '1500-2000', '2000+']
    df['customer_tenure_category'] = pd.cut(
        df['days_on_platform'],
        bins=bins,
        labels=labels,
        right=False
    )
    new_features.append('customer_tenure_category')
    
    # Add indicator columns for zero denominators
    df['has_365d_deliveries'] = (df['no_of_deliveries_last_365d'] > 0).astype(int)
    df['has_aov_history'] = (df['service_deliverd_aov_last_365d'] > 0).astype(int)
    new_features.extend(['has_365d_deliveries', 'has_aov_history'])
    
    return df, new_features


def create_dummy_variables(
    data: pd.DataFrame,
    categorical_features: List[str],
    categorical_engineered: List[str],
) -> pd.DataFrame:
    """
    Create dummy variables for categorical features with advanced name sanitization
    
    Args:
        data: Input DataFrame
        categorical_features: List of categorical column names to convert to dummies
        categorical_engineered: List of engineered categorical features
        
    Returns:
        pd.DataFrame: DataFrame with properly sanitized dummy variables
    """
    # Make a copy to avoid modifying original DataFrame
    processed_data = data.copy()
    
    # Create dummies for each categorical feature (with original names first)
    for feature in categorical_features + categorical_engineered:
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
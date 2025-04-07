"""
Data Processing Module

This module contains functions to process and transform raw data for the Lead Scoring model.
The code is adapted from the '02 - Processamento e tratamento dos dados.ipynb' notebook.
"""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def load_data(input_file):
    """
    Load data from a CSV file.
    
    Args:
        input_file (str): Path to the input CSV file.
        
    Returns:
        DataFrame: Loaded data.
    """
    # Check if path includes directory
    if os.path.dirname(input_file) == '':
        input_file = os.path.join('data', 'raw', input_file)
    
    return pd.read_csv(input_file)

def clean_data(df):
    """
    Clean the dataset by handling missing values, duplicates, and outliers.
    
    Args:
        df (DataFrame): Input dataframe.
        
    Returns:
        DataFrame: Cleaned dataframe.
    """
    # Make a copy to avoid modifying the original dataframe
    df_clean = df.copy()
    
    # Handle missing values
    df_clean = df_clean.fillna({
        'numeric_columns': df_clean.median(),
        'categorical_columns': 'Unknown'
    })
    
    # Remove duplicates
    df_clean = df_clean.drop_duplicates()
    
    # Handle outliers (example using IQR method)
    # This should be customized based on actual data
    for col in df_clean.select_dtypes(include=['float64', 'int64']).columns:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        df_clean[col] = np.where(
            (df_clean[col] < lower_bound) | (df_clean[col] > upper_bound),
            df_clean[col].median(),
            df_clean[col]
        )
    
    return df_clean

def feature_engineering(df):
    """
    Create new features from existing data.
    
    Args:
        df (DataFrame): Input dataframe.
        
    Returns:
        DataFrame: Dataframe with new features.
    """
    # Make a copy to avoid modifying the original dataframe
    df_featured = df.copy()
    
    # Example feature engineering (customize based on your needs)
    # Create binary flags
    # df_featured['has_previous_purchase'] = (df_featured['previous_purchases'] > 0).astype(int)
    
    # Create aggregated features
    # df_featured['total_spend'] = df_featured['amount_1'] + df_featured['amount_2']
    
    # Create ratio features
    # df_featured['conversion_ratio'] = df_featured['conversions'] / df_featured['visits']
    
    return df_featured

def encode_categorical(df):
    """
    Encode categorical variables using one-hot encoding.
    
    Args:
        df (DataFrame): Input dataframe.
        
    Returns:
        DataFrame: Dataframe with encoded categorical variables.
    """
    # Make a copy to avoid modifying the original dataframe
    df_encoded = df.copy()
    
    # Get categorical columns (excluding the target variable)
    cat_columns = df_encoded.select_dtypes(include=['object']).columns.tolist()
    if 'target' in cat_columns:
        cat_columns.remove('target')
    
    # One-hot encode categorical variables
    if cat_columns:
        df_encoded = pd.get_dummies(df_encoded, columns=cat_columns, drop_first=True)
    
    return df_encoded

def scale_features(df, target_column='target'):
    """
    Scale numerical features using StandardScaler.
    
    Args:
        df (DataFrame): Input dataframe.
        target_column (str): Name of the target column.
        
    Returns:
        DataFrame: Dataframe with scaled features.
        StandardScaler: Fitted scaler object for future transformations.
    """
    # Make a copy to avoid modifying the original dataframe
    df_scaled = df.copy()
    
    # Separate target variable
    X = df_scaled.drop(columns=[target_column])
    y = df_scaled[target_column]
    
    # Get numerical columns
    num_columns = X.select_dtypes(include=['float64', 'int64']).columns.tolist()
    
    if num_columns:
        # Scale numerical features
        scaler = StandardScaler()
        X[num_columns] = scaler.fit_transform(X[num_columns])
        
        # Combine features and target
        df_scaled = pd.concat([X, y], axis=1)
        
        return df_scaled, scaler
    
    return df_scaled, None

def split_train_test(df, target_column='target', test_size=0.2, random_state=42):
    """
    Split data into training and testing sets.
    
    Args:
        df (DataFrame): Input dataframe.
        target_column (str): Name of the target column.
        test_size (float): Proportion of the dataset to include in the test split.
        random_state (int): Random seed for reproducibility.
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

def process_data(input_file, output_file=None):
    """
    Process the data through the complete pipeline.
    
    Args:
        input_file (str): Path to the input CSV file.
        output_file (str): Path to the output CSV file.
        
    Returns:
        DataFrame: Processed dataframe.
    """
    # Load the data
    df = load_data(input_file)
    
    # Process the data
    df_clean = clean_data(df)
    df_featured = feature_engineering(df_clean)
    df_encoded = encode_categorical(df_featured)
    df_processed, _ = scale_features(df_encoded)
    
    # Save the processed data if output_file is specified
    if output_file:
        # Check if path includes directory
        if os.path.dirname(output_file) == '':
            output_dir = os.path.join('data', 'processed')
            os.makedirs(output_dir, exist_ok=True)
            output_file = os.path.join(output_dir, output_file)
        
        df_processed.to_csv(output_file, index=False)
    
    return df_processed

if __name__ == "__main__":
    # Example usage
    input_file = "data.csv"
    output_file = "data_processed.csv"
    
    processed_data = process_data(input_file, output_file)
    print(f"Data processing completed. Output saved to: {output_file}")
    print(f"Processed data shape: {processed_data.shape}") 
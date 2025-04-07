"""
Prediction Module

This module contains functions to make predictions using a trained model.
"""

import os
import sys
import joblib
import pandas as pd
import numpy as np

def load_model(model_path):
    """
    Load a trained model from disk.
    
    Args:
        model_path (str): Path to the model file.
        
    Returns:
        object: Loaded model.
    """
    return joblib.load(model_path)

def load_data(input_file):
    """
    Load data from a CSV file.
    
    Args:
        input_file (str): Path to the input CSV file.
        
    Returns:
        DataFrame: Loaded data.
    """
    return pd.read_csv(input_file)

def preprocess_data(df, target_column=None):
    """
    Preprocess data for prediction.
    
    Args:
        df (DataFrame): Input dataframe.
        target_column (str): Name of the target column, if any.
        
    Returns:
        DataFrame: Preprocessed features.
    """
    # Make a copy to avoid modifying the original dataframe
    df_copy = df.copy()
    
    # Handle missing values (example)
    for col in df_copy.select_dtypes(include=['float64', 'int64']).columns:
        df_copy[col].fillna(df_copy[col].median(), inplace=True)
    
    for col in df_copy.select_dtypes(include=['object']).columns:
        df_copy[col].fillna('Unknown', inplace=True)
    
    # Get features (if target is present, exclude it)
    if target_column and target_column in df_copy.columns:
        X = df_copy.drop(columns=[target_column])
        y = df_copy[target_column]
        return X, y
    else:
        return df_copy, None

def make_predictions(model, X):
    """
    Make predictions using a trained model.
    
    Args:
        model: Trained model.
        X (DataFrame): Features.
        
    Returns:
        tuple: (Predictions, Probabilities)
    """
    # Make predictions
    y_pred = model.predict(X)
    
    # Get probabilities (if available)
    if hasattr(model, 'predict_proba'):
        y_prob = model.predict_proba(X)[:, 1]
    else:
        y_prob = None
    
    return y_pred, y_prob

def save_predictions(df, y_pred, y_prob=None, output_file=None):
    """
    Save predictions to a CSV file.
    
    Args:
        df (DataFrame): Original dataframe.
        y_pred (array): Predictions.
        y_prob (array): Probabilities, if available.
        output_file (str): Path to the output CSV file.
        
    Returns:
        DataFrame: Dataframe with predictions.
    """
    # Make a copy to avoid modifying the original dataframe
    df_pred = df.copy()
    
    # Add predictions
    df_pred['prediction'] = y_pred
    
    # Add probabilities (if available)
    if y_prob is not None:
        df_pred['probability'] = y_prob
    
    # Save to CSV if output_file is specified
    if output_file:
        df_pred.to_csv(output_file, index=False)
        print(f"Predictions saved to: {output_file}")
    
    return df_pred

def predict(model_path, input_file, output_file=None, target_column=None):
    """
    Make predictions using a trained model.
    
    Args:
        model_path (str): Path to the model file.
        input_file (str): Path to the input CSV file.
        output_file (str): Path to the output CSV file.
        target_column (str): Name of the target column, if any.
        
    Returns:
        DataFrame: Dataframe with predictions.
    """
    # Load model
    model = load_model(model_path)
    
    # Load data
    df = load_data(input_file)
    
    # Preprocess data
    X, y = preprocess_data(df, target_column)
    
    # Make predictions
    y_pred, y_prob = make_predictions(model, X)
    
    # Calculate evaluation metrics if target is available
    if y is not None:
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
        
        metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred),
            'recall': recall_score(y, y_pred),
            'f1': f1_score(y, y_pred)
        }
        
        if y_prob is not None:
            metrics['auc'] = roc_auc_score(y, y_prob)
        
        print("\nEvaluation Metrics:")
        for metric, value in metrics.items():
            print(f"  {metric.capitalize()}: {value:.4f}")
    
    # Save predictions
    return save_predictions(df, y_pred, y_prob, output_file)

def main():
    """
    Main function to run the prediction script.
    """
    # Parse command line arguments
    if len(sys.argv) < 3:
        print("Usage: python predict.py <model_path> <input_file> [output_file] [target_column]")
        sys.exit(1)
    
    model_path = sys.argv[1]
    input_file = sys.argv[2]
    
    output_file = sys.argv[3] if len(sys.argv) > 3 else None
    target_column = sys.argv[4] if len(sys.argv) > 4 else None
    
    # Make predictions
    predict(model_path, input_file, output_file, target_column)

if __name__ == "__main__":
    main() 
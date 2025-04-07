"""
Training Module

This module contains functions to train machine learning models for lead scoring.
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from data_processing import load_data as load_raw_data, process_data
from modeling import train_models, evaluate_models, save_model, plot_feature_importance

def parse_args():
    """
    Parse command line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description='Train machine learning models for lead scoring.')
    
    parser.add_argument('--input-file', type=str, default='data/raw/data.csv',
                      help='Path to the input CSV file')
    parser.add_argument('--output-dir', type=str, default='models',
                      help='Directory to save trained models')
    parser.add_argument('--target-column', type=str, default='target',
                      help='Name of the target column')
    parser.add_argument('--test-size', type=float, default=0.2,
                      help='Proportion of the dataset to include in the test split')
    parser.add_argument('--random-state', type=int, default=42,
                      help='Random seed for reproducibility')
    parser.add_argument('--model-type', type=str, default='all',
                      choices=['random_forest', 'svm', 'naive_bayes', 'logistic_regression', 'knn', 'all'],
                      help='Type of model to train')
    
    return parser.parse_args()

def load_data(input_file, processed_file=None):
    """
    Load and process data for training.
    
    Args:
        input_file (str): Path to the input CSV file.
        processed_file (str): Path to save processed data.
        
    Returns:
        DataFrame: Processed data.
    """
    # If processed file exists, load it directly
    if processed_file and os.path.exists(processed_file):
        print(f"Loading processed data from: {processed_file}")
        return pd.read_csv(processed_file)
    
    # Otherwise, process the raw data
    print(f"Processing raw data from: {input_file}")
    return process_data(input_file, processed_file)

def prepare_data(df, target_column='target', test_size=0.2, random_state=42):
    """
    Prepare data for training by splitting into training and testing sets.
    
    Args:
        df (DataFrame): Input dataframe.
        target_column (str): Name of the target column.
        test_size (float): Proportion of the dataset to include in the test split.
        random_state (int): Random seed for reproducibility.
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    # Separate features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Testing set: {X_test.shape[0]} samples")
    
    return X_train, X_test, y_train, y_test

def get_models(model_type='all'):
    """
    Get models to train.
    
    Args:
        model_type (str): Type of model to train.
        
    Returns:
        dict: Dictionary of models to train.
    """
    models = {
        'random_forest': {'Random Forest': RandomForestClassifier(random_state=42)},
        'svm': {'SVM': SVC(probability=True, random_state=42)},
        'naive_bayes': {'Naive Bayes': GaussianNB()},
        'logistic_regression': {'Logistic Regression': LogisticRegression(random_state=42)},
        'knn': {'KNN': KNeighborsClassifier()}
    }
    
    if model_type == 'all':
        return {name: model for model_dict in models.values() for name, model in model_dict.items()}
    else:
        return models.get(model_type, {})

def train(args):
    """
    Train machine learning models.
    
    Args:
        args: Command line arguments.
        
    Returns:
        dict: Dictionary of evaluation metrics.
    """
    # Load and process data
    processed_file = os.path.join('data', 'processed', 'data_processed.csv')
    df = load_data(args.input_file, processed_file)
    
    # Prepare data
    X_train, X_test, y_train, y_test = prepare_data(
        df, args.target_column, args.test_size, args.random_state
    )
    
    # Get models to train
    models_to_train = get_models(args.model_type)
    
    if not models_to_train:
        print(f"Invalid model type: {args.model_type}")
        return {}
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    plots_dir = os.path.join(args.output_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # Train models
    print("\nTraining models...")
    trained_models, cv_scores = train_models(X_train, y_train, models_to_train)
    
    # Evaluate models
    print("\nEvaluating models...")
    results = evaluate_models(trained_models, X_test, y_test, plots_dir)
    
    # Find best model
    best_model_name = max(results, key=lambda x: results[x]['auc'])
    best_model = trained_models[best_model_name]
    
    print(f"\nBest model: {best_model_name}")
    print(f"AUC: {results[best_model_name]['auc']:.4f}")
    
    # Plot feature importance for tree-based models
    if 'Random Forest' in trained_models:
        plot_feature_importance(trained_models['Random Forest'], X_train, plots_dir)
    
    # Save all trained models
    for name, model in trained_models.items():
        save_model(model, name, args.output_dir)
    
    # Save best model separately
    save_model(best_model, 'best_model', args.output_dir)
    
    return results

def main():
    """
    Main function to run the training script.
    """
    # Parse command line arguments
    args = parse_args()
    
    # Train models
    train(args)

if __name__ == "__main__":
    main() 
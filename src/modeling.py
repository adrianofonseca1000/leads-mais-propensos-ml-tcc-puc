"""
Modeling Module

This module contains functions to train, evaluate, and compare different machine learning models.
The code is adapted from the '04 - Aplicação e avaliação de Modelos Machine Learning.ipynb' notebook.
"""

import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                            roc_auc_score, confusion_matrix, classification_report,
                            roc_curve, precision_recall_curve)
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

def load_data(input_file):
    """
    Load processed data from a CSV file.
    
    Args:
        input_file (str): Path to the input CSV file.
        
    Returns:
        DataFrame: Loaded data.
    """
    # Check if path includes directory
    if os.path.dirname(input_file) == '':
        input_file = os.path.join('data', 'processed', input_file)
    
    return pd.read_csv(input_file)

def prepare_data(df, target_column='target', test_size=0.2, random_state=42):
    """
    Prepare data for modeling by splitting into training and testing sets.
    
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
    
    return X_train, X_test, y_train, y_test

def train_models(X_train, y_train, models=None, cv=5):
    """
    Train multiple models using cross-validation.
    
    Args:
        X_train (DataFrame): Training features.
        y_train (Series): Training target.
        models (dict): Dictionary of models to train.
        cv (int): Number of cross-validation folds.
        
    Returns:
        dict: Dictionary of trained models.
    """
    # Define default models if none provided
    if models is None:
        models = {
            'Random Forest': RandomForestClassifier(random_state=42),
            'SVM': SVC(probability=True, random_state=42),
            'Naive Bayes': GaussianNB(),
            'Logistic Regression': LogisticRegression(random_state=42),
            'KNN': KNeighborsClassifier()
        }
    
    # Train models
    trained_models = {}
    cv_scores = {}
    
    for name, model in models.items():
        print(f"Training {name}...")
        
        # Perform cross-validation
        cv_score = cross_val_score(model, X_train, y_train, cv=cv, scoring='roc_auc')
        cv_scores[name] = cv_score
        
        # Train on full training set
        model.fit(X_train, y_train)
        trained_models[name] = model
        
        print(f"  CV ROC-AUC: {cv_score.mean():.4f} ± {cv_score.std():.4f}")
    
    return trained_models, cv_scores

def evaluate_models(models, X_test, y_test, output_dir=None):
    """
    Evaluate models on test data.
    
    Args:
        models (dict): Dictionary of trained models.
        X_test (DataFrame): Testing features.
        y_test (Series): Testing target.
        output_dir (str): Directory to save evaluation plots.
        
    Returns:
        dict: Dictionary of evaluation metrics.
    """
    # Create output directory if it doesn't exist
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Evaluate models
    results = {}
    
    for name, model in models.items():
        print(f"Evaluating {name}...")
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'auc': roc_auc_score(y_test, y_prob)
        }
        
        # Store results
        results[name] = metrics
        
        # Print metrics
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")
        print(f"  F1-score: {metrics['f1']:.4f}")
        print(f"  AUC: {metrics['auc']:.4f}")
        
        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
        
        plt.title(f'Confusion Matrix - {name}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        
        # Save plot if output_dir is specified
        if output_dir:
            plt.savefig(os.path.join(output_dir, f'confusion_matrix_{name.replace(" ", "_")}.png'))
            plt.close()
        else:
            plt.show()
        
        # Plot ROC curve
        plt.figure(figsize=(8, 6))
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        
        plt.plot(fpr, tpr, label=f'{name} (AUC = {metrics["auc"]:.4f})')
        plt.plot([0, 1], [0, 1], 'k--')
        
        plt.title(f'ROC Curve - {name}')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save plot if output_dir is specified
        if output_dir:
            plt.savefig(os.path.join(output_dir, f'roc_curve_{name.replace(" ", "_")}.png'))
            plt.close()
        else:
            plt.show()
    
    # Compare models
    plt.figure(figsize=(10, 8))
    
    metrics_df = pd.DataFrame(results).T
    metrics_df.plot(kind='bar', figsize=(12, 8))
    
    plt.title('Model Comparison')
    plt.xlabel('Model')
    plt.ylabel('Score')
    plt.ylim(0, 1)
    plt.xticks(rotation=45)
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    
    # Save plot if output_dir is specified
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'model_comparison.png'))
        plt.close()
    else:
        plt.show()
    
    return results

def tune_hyperparameters(X_train, y_train, model, param_grid, cv=5):
    """
    Tune hyperparameters using GridSearchCV.
    
    Args:
        X_train (DataFrame): Training features.
        y_train (Series): Training target.
        model: Base model to tune.
        param_grid (dict): Dictionary of hyperparameters to tune.
        cv (int): Number of cross-validation folds.
        
    Returns:
        tuple: (best_model, best_params, cv_results)
    """
    print(f"Tuning hyperparameters for {model.__class__.__name__}...")
    
    # Create GridSearchCV object
    grid_search = GridSearchCV(
        model, param_grid, cv=cv, scoring='roc_auc', n_jobs=-1, verbose=1
    )
    
    # Fit to data
    grid_search.fit(X_train, y_train)
    
    # Get best model and parameters
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    cv_results = grid_search.cv_results_
    
    print(f"Best parameters: {best_params}")
    print(f"Best CV score: {grid_search.best_score_:.4f}")
    
    return best_model, best_params, cv_results

def plot_feature_importance(model, X, output_dir=None):
    """
    Plot feature importance for tree-based models.
    
    Args:
        model: Trained model with feature_importances_ attribute.
        X (DataFrame): Feature dataframe.
        output_dir (str): Directory to save plots.
        
    Returns:
        None
    """
    # Check if model supports feature importance
    if not hasattr(model, 'feature_importances_'):
        print("Model does not support feature importance.")
        return
    
    # Get feature importance
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    # Plot feature importance
    plt.figure(figsize=(12, 8))
    
    plt.title('Feature Importance')
    plt.bar(range(X.shape[1]), importances[indices], align='center')
    plt.xticks(range(X.shape[1]), X.columns[indices], rotation=90)
    plt.xlim([-1, X.shape[1]])
    plt.tight_layout()
    
    # Save plot if output_dir is specified
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'feature_importance.png'))
        plt.close()
    else:
        plt.show()

def save_model(model, model_name, output_dir='models'):
    """
    Save model to disk.
    
    Args:
        model: Trained model to save.
        model_name (str): Name of the model.
        output_dir (str): Directory to save the model.
        
    Returns:
        str: Path to the saved model.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save model
    model_path = os.path.join(output_dir, f'{model_name.replace(" ", "_").lower()}.pkl')
    joblib.dump(model, model_path)
    
    print(f"Model saved to: {model_path}")
    
    return model_path

def train_and_evaluate(input_file, output_dir=None, target_column='target'):
    """
    Train and evaluate models.
    
    Args:
        input_file (str): Path to the input CSV file.
        output_dir (str): Directory to save models and plots.
        target_column (str): Name of the target column.
        
    Returns:
        dict: Dictionary of evaluation metrics.
    """
    # Load the data
    df = load_data(input_file)
    
    # Prepare data
    X_train, X_test, y_train, y_test = prepare_data(df, target_column)
    
    # Create output directories
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plots_dir = os.path.join(output_dir, 'plots')
        os.makedirs(plots_dir, exist_ok=True)
    else:
        plots_dir = None
    
    # Train models
    models, cv_scores = train_models(X_train, y_train)
    
    # Evaluate models
    results = evaluate_models(models, X_test, y_test, plots_dir)
    
    # Find best model
    best_model_name = max(results, key=lambda x: results[x]['auc'])
    best_model = models[best_model_name]
    
    print(f"\nBest model: {best_model_name}")
    print(f"AUC: {results[best_model_name]['auc']:.4f}")
    
    # Plot feature importance for tree-based models
    if best_model_name == 'Random Forest':
        plot_feature_importance(best_model, X_train, plots_dir)
    
    # Save best model
    if output_dir:
        save_model(best_model, best_model_name, output_dir)
    
    return results

if __name__ == "__main__":
    # Example usage
    input_file = "data_processed.csv"
    output_dir = "models"
    
    results = train_and_evaluate(input_file, output_dir) 
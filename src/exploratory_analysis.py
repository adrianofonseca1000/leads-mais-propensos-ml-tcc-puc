"""
Exploratory Data Analysis Module

This module contains functions to perform exploratory data analysis on the processed data.
The code is adapted from the '03 - Análise e exploração dos dados.ipynb' notebook.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

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

def data_summary(df):
    """
    Generate summary statistics for the dataset.
    
    Args:
        df (DataFrame): Input dataframe.
        
    Returns:
        dict: Dictionary containing summary statistics.
    """
    summary = {
        'shape': df.shape,
        'dtypes': df.dtypes,
        'describe': df.describe(),
        'missing_values': df.isnull().sum(),
        'unique_values': {col: df[col].nunique() for col in df.columns}
    }
    
    return summary

def plot_distributions(df, target_column='target', output_dir=None):
    """
    Plot distributions of numerical features.
    
    Args:
        df (DataFrame): Input dataframe.
        target_column (str): Name of the target column.
        output_dir (str): Directory to save plots.
        
    Returns:
        None
    """
    # Create output directory if it doesn't exist
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Get numerical columns
    num_columns = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    if target_column in num_columns:
        num_columns.remove(target_column)
    
    # Plot distributions
    for col in num_columns:
        plt.figure(figsize=(10, 6))
        
        # Plot distribution by target
        sns.histplot(data=df, x=col, hue=target_column, kde=True, element='step')
        
        plt.title(f'Distribution of {col} by {target_column}')
        plt.xlabel(col)
        plt.ylabel('Frequency')
        
        # Save plot if output_dir is specified
        if output_dir:
            plt.savefig(os.path.join(output_dir, f'distribution_{col}.png'))
            plt.close()
        else:
            plt.show()

def plot_correlations(df, target_column='target', output_dir=None):
    """
    Plot correlation matrix and correlations with target.
    
    Args:
        df (DataFrame): Input dataframe.
        target_column (str): Name of the target column.
        output_dir (str): Directory to save plots.
        
    Returns:
        None
    """
    # Create output directory if it doesn't exist
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Get numerical columns
    num_columns = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    
    # Plot correlation matrix
    plt.figure(figsize=(12, 10))
    corr_matrix = df[num_columns].corr()
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', linewidths=0.5)
    
    plt.title('Correlation Matrix')
    plt.tight_layout()
    
    # Save plot if output_dir is specified
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'correlation_matrix.png'))
        plt.close()
    else:
        plt.show()
    
    # Plot correlations with target
    if target_column in df.columns:
        plt.figure(figsize=(10, 8))
        target_corr = corr_matrix[target_column].sort_values(ascending=False)
        target_corr = target_corr.drop(target_column)
        
        sns.barplot(x=target_corr.values, y=target_corr.index)
        
        plt.title(f'Correlations with {target_column}')
        plt.xlabel('Correlation')
        plt.tight_layout()
        
        # Save plot if output_dir is specified
        if output_dir:
            plt.savefig(os.path.join(output_dir, f'correlations_with_{target_column}.png'))
            plt.close()
        else:
            plt.show()

def perform_pca(df, target_column='target', n_components=2, output_dir=None):
    """
    Perform PCA and visualize the results.
    
    Args:
        df (DataFrame): Input dataframe.
        target_column (str): Name of the target column.
        n_components (int): Number of principal components.
        output_dir (str): Directory to save plots.
        
    Returns:
        DataFrame: DataFrame with PCA components.
    """
    # Create output directory if it doesn't exist
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Separate features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # Get numerical columns
    num_columns = X.select_dtypes(include=['float64', 'int64']).columns.tolist()
    
    # Scale the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X[num_columns])
    
    # Perform PCA
    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(X_scaled)
    
    # Create PCA DataFrame
    pca_df = pd.DataFrame(data=principal_components, 
                          columns=[f'PC{i+1}' for i in range(n_components)])
    pca_df[target_column] = y.values
    
    # Plot PCA
    if n_components >= 2:
        plt.figure(figsize=(10, 8))
        
        # Plot by target
        sns.scatterplot(data=pca_df, x='PC1', y='PC2', hue=target_column, palette='viridis')
        
        plt.title('PCA of Lead Data')
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
        
        # Save plot if output_dir is specified
        if output_dir:
            plt.savefig(os.path.join(output_dir, 'pca_visualization.png'))
            plt.close()
        else:
            plt.show()
    
    # Plot explained variance
    plt.figure(figsize=(10, 6))
    
    plt.plot(np.cumsum(pca.explained_variance_ratio_), marker='o')
    
    plt.title('Explained Variance by Components')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.grid(True)
    
    # Save plot if output_dir is specified
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'explained_variance.png'))
        plt.close()
    else:
        plt.show()
    
    return pca_df

def create_visualizations(input_file, output_dir=None):
    """
    Create visualizations for exploratory data analysis.
    
    Args:
        input_file (str): Path to the input CSV file.
        output_dir (str): Directory to save plots.
        
    Returns:
        None
    """
    # Load the data
    df = load_data(input_file)
    
    # Create output directory if it doesn't exist
    if output_dir:
        output_dir = os.path.join('reports', 'figures', output_dir)
        os.makedirs(output_dir, exist_ok=True)
    
    # Generate summary statistics
    summary = data_summary(df)
    print("Summary Statistics:")
    print(f"Dataset shape: {summary['shape']}")
    print(f"Missing values: {summary['missing_values'].sum()}")
    
    # Create visualizations
    plot_distributions(df, output_dir=output_dir)
    plot_correlations(df, output_dir=output_dir)
    perform_pca(df, output_dir=output_dir)
    
    print(f"Visualizations created and saved to: {output_dir}")

if __name__ == "__main__":
    # Example usage
    input_file = "data_processed.csv"
    output_dir = "eda_visualizations"
    
    create_visualizations(input_file, output_dir) 
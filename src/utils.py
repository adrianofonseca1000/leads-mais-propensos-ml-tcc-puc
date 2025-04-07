"""
Utility Module

This module contains utility functions shared across the project.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import logging

# Configure logging
def setup_logging(log_file=None, level=logging.INFO):
    """
    Set up logging configuration.
    
    Args:
        log_file (str): Path to the log file.
        level: Logging level.
        
    Returns:
        logging.Logger: Configured logger.
    """
    # Create logger
    logger = logging.getLogger('lead_scoring')
    logger.setLevel(level)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    
    # Add console handler to logger
    logger.addHandler(console_handler)
    
    # Add file handler if log_file is specified
    if log_file:
        # Create logs directory if it doesn't exist
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        
        # Create file handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        
        # Add file handler to logger
        logger.addHandler(file_handler)
    
    return logger

# File utilities
def ensure_dir(directory):
    """
    Create directory if it doesn't exist.
    
    Args:
        directory (str): Directory path.
        
    Returns:
        str: Directory path.
    """
    os.makedirs(directory, exist_ok=True)
    return directory

def get_timestamp():
    """
    Get current timestamp as string.
    
    Returns:
        str: Current timestamp in format YYYYMMDD_HHMMSS.
    """
    return datetime.now().strftime('%Y%m%d_%H%M%S')

def save_dataframe(df, filepath, index=False):
    """
    Save DataFrame to CSV.
    
    Args:
        df (DataFrame): DataFrame to save.
        filepath (str): Path to save the DataFrame.
        index (bool): Whether to include index in the CSV.
        
    Returns:
        str: Path to the saved DataFrame.
    """
    # Create directory if it doesn't exist
    directory = os.path.dirname(filepath)
    if directory:
        os.makedirs(directory, exist_ok=True)
    
    # Save DataFrame
    df.to_csv(filepath, index=index)
    
    return filepath

# Data analysis utilities
def describe_dataframe(df):
    """
    Generate comprehensive description of DataFrame.
    
    Args:
        df (DataFrame): Input DataFrame.
        
    Returns:
        dict: Dictionary with DataFrame description.
    """
    description = {
        'shape': df.shape,
        'columns': df.columns.tolist(),
        'dtypes': df.dtypes,
        'describe': df.describe(),
        'missing_values': df.isnull().sum(),
        'missing_percent': (df.isnull().sum() / len(df)) * 100,
        'unique_values': {col: df[col].nunique() for col in df.columns}
    }
    
    return description

def memory_usage(df):
    """
    Calculate memory usage of DataFrame.
    
    Args:
        df (DataFrame): Input DataFrame.
        
    Returns:
        tuple: (Total memory in MB, Memory by column in MB)
    """
    memory_by_column = df.memory_usage(deep=True) / (1024 * 1024)  # MB
    total_memory = memory_by_column.sum()
    
    return total_memory, memory_by_column

def reduce_memory_usage(df):
    """
    Reduce memory usage of DataFrame by optimizing data types.
    
    Args:
        df (DataFrame): Input DataFrame.
        
    Returns:
        DataFrame: DataFrame with optimized data types.
    """
    # Make a copy to avoid modifying the original dataframe
    df_optimized = df.copy()
    
    # Process columns by data type
    for col in df_optimized.columns:
        # Skip non-numeric columns
        if df_optimized[col].dtype == 'object':
            continue
        
        # Convert integer columns
        if pd.api.types.is_integer_dtype(df_optimized[col].dtype):
            min_val = df_optimized[col].min()
            max_val = df_optimized[col].max()
            
            if min_val >= 0:
                if max_val < 2**8:
                    df_optimized[col] = df_optimized[col].astype(np.uint8)
                elif max_val < 2**16:
                    df_optimized[col] = df_optimized[col].astype(np.uint16)
                elif max_val < 2**32:
                    df_optimized[col] = df_optimized[col].astype(np.uint32)
                else:
                    df_optimized[col] = df_optimized[col].astype(np.uint64)
            else:
                if min_val > -2**7 and max_val < 2**7:
                    df_optimized[col] = df_optimized[col].astype(np.int8)
                elif min_val > -2**15 and max_val < 2**15:
                    df_optimized[col] = df_optimized[col].astype(np.int16)
                elif min_val > -2**31 and max_val < 2**31:
                    df_optimized[col] = df_optimized[col].astype(np.int32)
                else:
                    df_optimized[col] = df_optimized[col].astype(np.int64)
        
        # Convert float columns
        elif pd.api.types.is_float_dtype(df_optimized[col].dtype):
            # Check if column contains mostly integers with some NaNs
            if df_optimized[col].isnull().sum() > 0 and df_optimized[col].dropna().apply(lambda x: float(x).is_integer()).all():
                # Use smallest int type that accommodates NaN
                df_optimized[col] = pd.to_numeric(df_optimized[col], downcast='float')
            else:
                # Use smallest float type
                df_optimized[col] = pd.to_numeric(df_optimized[col], downcast='float')
    
    return df_optimized

# Plotting utilities
def set_plot_style():
    """
    Set common matplotlib plot style.
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12
    plt.rcParams['legend.fontsize'] = 12

def save_plot(filepath, dpi=300, tight=True):
    """
    Save current plot to file.
    
    Args:
        filepath (str): Path to save the plot.
        dpi (int): DPI for the output image.
        tight (bool): Whether to use tight layout.
        
    Returns:
        str: Path to the saved plot.
    """
    # Create directory if it doesn't exist
    directory = os.path.dirname(filepath)
    if directory:
        os.makedirs(directory, exist_ok=True)
    
    # Save plot
    if tight:
        plt.tight_layout()
    
    plt.savefig(filepath, dpi=dpi, bbox_inches='tight')
    plt.close()
    
    return filepath

# Path utilities
def get_project_root():
    """
    Get the project root directory.
    
    Returns:
        str: Project root directory.
    """
    # Assuming this file is in the src directory of the project
    return os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

def get_data_path(data_type='raw'):
    """
    Get the path to the data directory.
    
    Args:
        data_type (str): Type of data (raw, processed, etc.).
        
    Returns:
        str: Path to the data directory.
    """
    project_root = get_project_root()
    data_path = os.path.join(project_root, 'data', data_type)
    os.makedirs(data_path, exist_ok=True)
    
    return data_path

def get_models_path():
    """
    Get the path to the models directory.
    
    Returns:
        str: Path to the models directory.
    """
    project_root = get_project_root()
    models_path = os.path.join(project_root, 'models')
    os.makedirs(models_path, exist_ok=True)
    
    return models_path

def get_reports_path():
    """
    Get the path to the reports directory.
    
    Returns:
        str: Path to the reports directory.
    """
    project_root = get_project_root()
    reports_path = os.path.join(project_root, 'reports')
    os.makedirs(reports_path, exist_ok=True)
    
    return reports_path 
"""
Data processing module for the Lead Scoring project.
Implements the DataProcessor interface for cleaning and preparing data.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
import logging

from ..config import CATEGORICAL_COLUMNS, NUMERICAL_COLUMNS, TARGET_COLUMN

class DataProcessor(ABC):
    """Abstract base class for data processors."""
    
    @abstractmethod
    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process the input dataframe."""
        pass

class LeadScoringDataProcessor(DataProcessor):
    """Concrete implementation of DataProcessor for lead scoring data."""
    
    def __init__(self):
        """Initialize the lead scoring data processor."""
        self.logger = logging.getLogger(__name__)
    
    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process the input dataframe.
        
        Args:
            df (pd.DataFrame): Input dataframe to process.
            
        Returns:
            pd.DataFrame: Processed dataframe.
        """
        self.logger.info("Starting data processing")
        
        # Make a copy to avoid modifying the original dataframe
        df_processed = df.copy()
        
        # Handle missing values
        df_processed = self._handle_missing_values(df_processed)
        
        # Remove duplicates
        df_processed = self._remove_duplicates(df_processed)
        
        # Handle outliers
        df_processed = self._handle_outliers(df_processed)
        
        # Do a final check for missing values and fill them if any remain
        df_processed = self._final_cleanup(df_processed)
        
        # Validate the processed data
        self._validate_processed_data(df_processed)
        
        self.logger.info("Data processing completed")
        return df_processed
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the dataframe."""
        self.logger.info("Handling missing values")
        
        # Check total missing values
        total_missing = df.isnull().sum().sum()
        self.logger.info(f"Total missing values: {total_missing}")
        
        # Fill numerical columns with median
        for col in df.select_dtypes(include=['float64', 'int64']).columns:
            missing = df[col].isnull().sum()
            if missing > 0:
                self.logger.info(f"Filling {missing} missing values in column {col} with median")
                df[col] = df[col].fillna(df[col].median())
        
        # Fill categorical columns with mode
        for col in df.select_dtypes(include=['object']).columns:
            missing = df[col].isnull().sum()
            if missing > 0:
                if len(df[col].dropna()) > 0:
                    mode_value = df[col].mode()[0]
                    self.logger.info(f"Filling {missing} missing values in column {col} with mode: {mode_value}")
                    df[col] = df[col].fillna(mode_value)
                else:
                    self.logger.warning(f"No non-null values in column {col}, filling with 'Unknown'")
                    df[col] = df[col].fillna('Unknown')
        
        return df
    
    def _remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate rows from the dataframe."""
        self.logger.info("Removing duplicates")
        initial_rows = len(df)
        df = df.drop_duplicates()
        removed_rows = initial_rows - len(df)
        if removed_rows > 0:
            self.logger.warning(f"Removed {removed_rows} duplicate rows")
        return df
    
    def _handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle outliers in numerical columns using IQR method."""
        self.logger.info("Handling outliers")
        
        for col in df.select_dtypes(include=['float64', 'int64']).columns:
            # Skip columns that are all nulls or have only one unique value
            if df[col].isnull().all() or df[col].nunique() <= 1:
                continue
                
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
            if outliers > 0:
                self.logger.warning(f"Found {outliers} outliers in column {col}")
                
                # Replace outliers with median
                median_value = df[col].median()
                df[col] = np.where(
                    (df[col] < lower_bound) | (df[col] > upper_bound),
                    median_value,
                    df[col]
                )
        
        return df
    
    def _final_cleanup(self, df: pd.DataFrame) -> pd.DataFrame:
        """Perform final cleanup to ensure no missing values remain."""
        self.logger.info("Performing final cleanup")
        
        # Check remaining missing values
        missing_values = df.isnull().sum().sum()
        if missing_values > 0:
            self.logger.warning(f"Found {missing_values} missing values after initial processing, applying final cleanup")
            
            # For any remaining numeric columns with nulls, fill with 0
            for col in df.select_dtypes(include=['float64', 'int64']).columns:
                if df[col].isnull().any():
                    self.logger.warning(f"Filling remaining nulls in numeric column {col} with 0")
                    df[col] = df[col].fillna(0)
            
            # For any remaining categorical columns with nulls, fill with 'Unknown'
            for col in df.select_dtypes(include=['object']).columns:
                if df[col].isnull().any():
                    self.logger.warning(f"Filling remaining nulls in categorical column {col} with 'Unknown'")
                    df[col] = df[col].fillna('Unknown')
        
        return df
    
    def _validate_processed_data(self, df: pd.DataFrame) -> None:
        """Validate the processed dataframe."""
        self.logger.info("Validating processed data")
        
        # Check for remaining missing values
        missing_values = df.isnull().sum().sum()
        if missing_values > 0:
            self.logger.error(f"Found {missing_values} missing values after processing")
            
            # Log columns with missing values
            missing_cols = df.columns[df.isnull().any()].tolist()
            self.logger.error(f"Columns with missing values: {missing_cols}")
            
            # Get counts of missing values by column
            missing_counts = df.isnull().sum()
            missing_counts = missing_counts[missing_counts > 0]
            for col, count in missing_counts.items():
                self.logger.error(f"Column {col}: {count} missing values")
                
            raise ValueError("Data processing failed: Missing values remain")
        
        # Check for required columns
        missing_columns = []
        for col in CATEGORICAL_COLUMNS + NUMERICAL_COLUMNS + [TARGET_COLUMN]:
            if col not in df.columns:
                missing_columns.append(col)
        
        if missing_columns:
            self.logger.error(f"Missing required columns: {missing_columns}")
            raise ValueError(f"Data processing failed: Missing required columns: {missing_columns}")

class DataProcessorFactory:
    """Factory class for creating data processors."""
    
    @staticmethod
    def create_processor(processor_type: str = 'lead_scoring', **kwargs) -> DataProcessor:
        """
        Create a data processor instance.
        
        Args:
            processor_type (str): Type of processor to create. Currently only 'lead_scoring' is supported.
            **kwargs: Additional arguments to pass to the processor constructor.
            
        Returns:
            DataProcessor: A data processor instance.
            
        Raises:
            ValueError: If the processor type is not supported.
        """
        if processor_type.lower() == 'lead_scoring':
            return LeadScoringDataProcessor(**kwargs)
        else:
            raise ValueError(f"Unsupported processor type: {processor_type}")

# Default processor instance
default_processor = DataProcessorFactory.create_processor() 
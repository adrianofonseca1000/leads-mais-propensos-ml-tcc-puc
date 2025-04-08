"""
Data validation module for the Lead Scoring project.
Implements the DataValidator interface for validating data at different stages.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
import logging

from ..config import CATEGORICAL_COLUMNS, NUMERICAL_COLUMNS, TARGET_COLUMN

class DataValidator(ABC):
    """Abstract base class for data validators."""
    
    @abstractmethod
    def validate(self, df: pd.DataFrame, stage: str) -> bool:
        """Validate the dataframe at the specified stage."""
        pass

class LeadScoringDataValidator(DataValidator):
    """Concrete implementation of DataValidator for lead scoring data."""
    
    def __init__(self):
        """Initialize the lead scoring data validator."""
        self.logger = logging.getLogger(__name__)
    
    def validate(self, df: pd.DataFrame, stage: str) -> bool:
        """
        Validate the dataframe at the specified stage.
        
        Args:
            df (pd.DataFrame): Dataframe to validate.
            stage (str): Stage of validation ('raw', 'processed', 'transformed').
            
        Returns:
            bool: True if validation passes, False otherwise.
        """
        self.logger.info(f"Starting {stage} data validation")
        
        try:
            if stage == 'raw':
                return self._validate_raw_data(df)
            elif stage == 'processed':
                return self._validate_processed_data(df)
            elif stage == 'transformed':
                return self._validate_transformed_data(df)
            else:
                raise ValueError(f"Unknown validation stage: {stage}")
        except Exception as e:
            self.logger.error(f"Validation failed at {stage} stage: {str(e)}")
            return False
    
    def _validate_raw_data(self, df: pd.DataFrame) -> bool:
        """Validate raw data."""
        # Check if dataframe is not empty
        if df.empty:
            self.logger.error("Raw data is empty")
            return False
        
        # Check for required columns - modified to be more flexible
        # For raw data, we'll only check for the target column
        if TARGET_COLUMN not in df.columns:
            self.logger.error(f"Missing target column: {TARGET_COLUMN}")
            return False
        
        # Log which of the recommended columns are present
        present_cat_cols = [col for col in CATEGORICAL_COLUMNS if col in df.columns]
        present_num_cols = [col for col in NUMERICAL_COLUMNS if col in df.columns]
        
        missing_cat_cols = [col for col in CATEGORICAL_COLUMNS if col not in df.columns]
        missing_num_cols = [col for col in NUMERICAL_COLUMNS if col not in df.columns]
        
        if missing_cat_cols or missing_num_cols:
            self.logger.warning(f"Missing some recommended columns: "
                               f"Categorical: {missing_cat_cols}, "
                               f"Numerical: {missing_num_cols}")
        
        self.logger.info(f"Found {len(present_cat_cols)} categorical columns and "
                        f"{len(present_num_cols)} numerical columns")
        
        # Check for data types - only for columns that are present
        for col in present_cat_cols:
            if not pd.api.types.is_object_dtype(df[col]) and not pd.api.types.is_categorical_dtype(df[col]):
                self.logger.warning(f"Column {col} is in CATEGORICAL_COLUMNS but is not categorical type")
        
        for col in present_num_cols:
            if not pd.api.types.is_numeric_dtype(df[col]):
                self.logger.warning(f"Column {col} is in NUMERICAL_COLUMNS but is not numeric type")
        
        return True
    
    def _validate_processed_data(self, df: pd.DataFrame) -> bool:
        """Validate processed data."""
        # Check for missing values
        missing_values = df.isnull().sum().sum()
        if missing_values > 0:
            self.logger.error(f"Processed data contains {missing_values} missing values")
            
            # Print which columns have missing values
            cols_with_missing = df.columns[df.isnull().any()].tolist()
            self.logger.error(f"Columns with missing values: {cols_with_missing}")
            
            return False
        
        # Check for duplicates
        duplicates = df.duplicated().sum()
        if duplicates > 0:
            self.logger.warning(f"Processed data contains {duplicates} duplicates")
        
        # Check for outliers in numerical columns
        for col in df.select_dtypes(include=['float64', 'int64']).columns:
            # Skip columns with mostly zeros or only one unique value
            if (df[col] == 0).mean() > 0.9 or df[col].nunique() <= 1:
                continue
                
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
            if outliers > 0:
                self.logger.warning(f"Found {outliers} outliers in column {col}")
        
        return True
    
    def _validate_transformed_data(self, df: pd.DataFrame) -> bool:
        """Validate transformed data."""
        # Check for infinite values
        inf_values = np.isinf(df.select_dtypes(include=['float64', 'int64']).values).sum()
        if inf_values > 0:
            self.logger.error(f"Transformed data contains {inf_values} infinite values")
            return False
        
        # Check for NaN values
        missing_values = df.isnull().sum().sum()
        if missing_values > 0:
            self.logger.error(f"Transformed data contains {missing_values} NaN values")
            return False
        
        # Check for negative values in numerical columns where it doesn't make sense
        for col in df.select_dtypes(include=['float64', 'int64']).columns:
            # Skip columns that might legitimately have negative values (like ratios)
            if 'ratio' in col.lower() or 'diff' in col.lower() or 'change' in col.lower():
                continue
                
            neg_values = (df[col] < 0).sum()
            if neg_values > 0:
                self.logger.warning(f"Column {col} contains {neg_values} negative values after transformation")
        
        return True

class DataValidatorFactory:
    """Factory class for creating data validators."""
    
    @staticmethod
    def create_validator(validator_type: str = 'lead_scoring', **kwargs) -> DataValidator:
        """
        Create a data validator instance.
        
        Args:
            validator_type (str): Type of validator to create. Currently only 'lead_scoring' is supported.
            **kwargs: Additional arguments to pass to the validator constructor.
            
        Returns:
            DataValidator: A data validator instance.
            
        Raises:
            ValueError: If the validator type is not supported.
        """
        if validator_type.lower() == 'lead_scoring':
            return LeadScoringDataValidator(**kwargs)
        else:
            raise ValueError(f"Unsupported validator type: {validator_type}")

# Default validator instance
default_validator = DataValidatorFactory.create_validator() 
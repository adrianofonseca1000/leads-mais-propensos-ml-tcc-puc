"""
Data transformation module for the Lead Scoring project.
Implements the DataTransformer interface for feature engineering and encoding.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import logging

from ..config import CATEGORICAL_COLUMNS, NUMERICAL_COLUMNS, TARGET_COLUMN

class DataTransformer(ABC):
    """Abstract base class for data transformers."""
    
    @abstractmethod
    def transform(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Transform the input dataframe and return the transformed data and transformers."""
        pass

class LeadScoringDataTransformer(DataTransformer):
    """Concrete implementation of DataTransformer for lead scoring data."""
    
    def __init__(self):
        """Initialize the lead scoring data transformer."""
        self.logger = logging.getLogger(__name__)
        self.scaler = StandardScaler()
        self.label_encoders = {}
    
    def transform(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Transform the input dataframe.
        
        Args:
            df (pd.DataFrame): Input dataframe to transform.
            
        Returns:
            Tuple[pd.DataFrame, Dict[str, Any]]: Transformed dataframe and transformers.
        """
        self.logger.info("Starting data transformation")
        
        # Make a copy to avoid modifying the original dataframe
        df_transformed = df.copy()
        
        # Encode categorical variables
        df_transformed = self._encode_categorical(df_transformed)
        
        # Scale numerical features
        df_transformed = self._scale_numerical(df_transformed)
        
        # Create new features
        df_transformed = self._create_features(df_transformed)
        
        # Validate the transformed data
        self._validate_transformed_data(df_transformed)
        
        self.logger.info("Data transformation completed")
        
        # Return transformed data and transformers
        transformers = {
            'scaler': self.scaler,
            'label_encoders': self.label_encoders
        }
        
        return df_transformed, transformers
    
    def _encode_categorical(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical variables using label encoding."""
        self.logger.info("Encoding categorical variables")
        
        for col in CATEGORICAL_COLUMNS:
            if col in df.columns:
                self.label_encoders[col] = LabelEncoder()
                df[col] = self.label_encoders[col].fit_transform(df[col])
        
        return df
    
    def _scale_numerical(self, df: pd.DataFrame) -> pd.DataFrame:
        """Scale numerical features using StandardScaler."""
        self.logger.info("Scaling numerical features")
        
        if NUMERICAL_COLUMNS:
            df[NUMERICAL_COLUMNS] = self.scaler.fit_transform(df[NUMERICAL_COLUMNS])
        
        return df
    
    def _create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create new features from existing ones."""
        self.logger.info("Creating new features")
        
        # Example feature engineering
        if 'idade_cliente' in df.columns and 'Qnt_abandono' in df.columns:
            df['idade_abandono_ratio'] = df['idade_cliente'] / (df['Qnt_abandono'] + 1)
        
        if 'sum_recharge' in df.columns and 'recharge_frequency' in df.columns:
            df['avg_recharge_amount'] = df['sum_recharge'] / (df['recharge_frequency'] + 1)
        
        return df
    
    def _validate_transformed_data(self, df: pd.DataFrame) -> None:
        """Validate the transformed dataframe."""
        self.logger.info("Validating transformed data")
        
        # Check for infinite values
        if np.isinf(df.values).any():
            self.logger.error("Found infinite values after transformation")
            raise ValueError("Data transformation failed: Infinite values present")
        
        # Check for NaN values
        if df.isnull().sum().sum() > 0:
            self.logger.error("Found NaN values after transformation")
            raise ValueError("Data transformation failed: NaN values present")

class DataTransformerFactory:
    """Factory class for creating data transformers."""
    
    @staticmethod
    def create_transformer(transformer_type: str = 'lead_scoring', **kwargs) -> DataTransformer:
        """
        Create a data transformer instance.
        
        Args:
            transformer_type (str): Type of transformer to create. Currently only 'lead_scoring' is supported.
            **kwargs: Additional arguments to pass to the transformer constructor.
            
        Returns:
            DataTransformer: A data transformer instance.
            
        Raises:
            ValueError: If the transformer type is not supported.
        """
        if transformer_type.lower() == 'lead_scoring':
            return LeadScoringDataTransformer(**kwargs)
        else:
            raise ValueError(f"Unsupported transformer type: {transformer_type}")

# Default transformer instance
default_transformer = DataTransformerFactory.create_transformer() 
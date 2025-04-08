"""
Data loading module for the Lead Scoring project.
Implements the DataLoader interface for loading data from various sources.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Union
import pandas as pd
import logging

from ..config import RAW_DATA_FILE, INTERMEDIATE_DATA_FILE, FINAL_DATA_FILE

class DataLoader(ABC):
    """Abstract base class for data loaders."""
    
    @abstractmethod
    def load(self, file_path: Union[str, Path]) -> pd.DataFrame:
        """Load data from the specified file path."""
        pass

class CSVDataLoader(DataLoader):
    """Concrete implementation of DataLoader for CSV files."""
    
    def __init__(self, sep: str = ';'):
        """
        Initialize the CSV data loader.
        
        Args:
            sep (str): CSV separator character. Defaults to ';'.
        """
        self.sep = sep
        self.logger = logging.getLogger(__name__)
    
    def load(self, file_path: Union[str, Path]) -> pd.DataFrame:
        """
        Load data from a CSV file.
        
        Args:
            file_path (Union[str, Path]): Path to the CSV file.
            
        Returns:
            pd.DataFrame: Loaded data.
            
        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file cannot be read as a CSV.
        """
        try:
            self.logger.info(f"Loading data from {file_path}")
            df = pd.read_csv(file_path, sep=self.sep)
            self.logger.info(f"Successfully loaded {len(df)} rows")
            return df
        except FileNotFoundError:
            self.logger.error(f"File not found: {file_path}")
            raise
        except Exception as e:
            self.logger.error(f"Error loading data: {str(e)}")
            raise ValueError(f"Could not load data from {file_path}: {str(e)}")

class DataLoaderFactory:
    """Factory class for creating data loaders."""
    
    @staticmethod
    def create_loader(loader_type: str = 'csv', **kwargs) -> DataLoader:
        """
        Create a data loader instance.
        
        Args:
            loader_type (str): Type of loader to create. Currently only 'csv' is supported.
            **kwargs: Additional arguments to pass to the loader constructor.
            
        Returns:
            DataLoader: A data loader instance.
            
        Raises:
            ValueError: If the loader type is not supported.
        """
        if loader_type.lower() == 'csv':
            return CSVDataLoader(**kwargs)
        else:
            raise ValueError(f"Unsupported loader type: {loader_type}")

# Default loader instance
default_loader = DataLoaderFactory.create_loader() 
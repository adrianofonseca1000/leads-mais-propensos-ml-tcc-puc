"""
Data pipeline module for the Lead Scoring project.
Orchestrates the data processing flow from raw data to transformed data.
"""

from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
import logging
from pathlib import Path

from .loader import DataLoader, DataLoaderFactory
from .processor import DataProcessor, DataProcessorFactory
from .transformer import DataTransformer, DataTransformerFactory
from .validator import DataValidator, DataValidatorFactory
from ..config import (
    RAW_DATA_FILE,
    INTERMEDIATE_DATA_FILE,
    FINAL_DATA_FILE,
    RANDOM_SEED
)

class DataPipeline:
    """Main data pipeline class that orchestrates the data processing flow."""
    
    def __init__(
        self,
        loader: Optional[DataLoader] = None,
        processor: Optional[DataProcessor] = None,
        transformer: Optional[DataTransformer] = None,
        validator: Optional[DataValidator] = None
    ):
        """
        Initialize the data pipeline.
        
        Args:
            loader: Data loader instance. If None, a default loader will be created.
            processor: Data processor instance. If None, a default processor will be created.
            transformer: Data transformer instance. If None, a default transformer will be created.
            validator: Data validator instance. If None, a default validator will be created.
        """
        self.logger = logging.getLogger(__name__)
        
        # Initialize components with defaults if not provided
        self.loader = loader or DataLoaderFactory.create_loader()
        self.processor = processor or DataProcessorFactory.create_processor()
        self.transformer = transformer or DataTransformerFactory.create_transformer()
        self.validator = validator or DataValidatorFactory.create_validator()
        
        # Set random seed for reproducibility
        pd.set_option('mode.chained_assignment', None)
        np.random.seed(RANDOM_SEED)
    
    def run(self, input_file: Path, output_file: Path) -> Dict[str, Any]:
        """
        Run the complete data pipeline.
        
        Args:
            input_file: Path to the input data file.
            output_file: Path to save the processed data.
            
        Returns:
            Dict[str, Any]: Dictionary containing the processed data and transformers.
        """
        self.logger.info("Starting data pipeline")
        
        try:
            # Load data
            self.logger.info("Loading data")
            df = self.loader.load(input_file)
            
            # Validate raw data
            if not self.validator.validate(df, 'raw'):
                raise ValueError("Raw data validation failed")
            
            # Process data
            self.logger.info("Processing data")
            df_processed = self.processor.process(df)
            
            # Validate processed data
            if not self.validator.validate(df_processed, 'processed'):
                raise ValueError("Processed data validation failed")
            
            # Transform data
            self.logger.info("Transforming data")
            df_transformed, transformers = self.transformer.transform(df_processed)
            
            # Validate transformed data
            if not self.validator.validate(df_transformed, 'transformed'):
                raise ValueError("Transformed data validation failed")
            
            # Save processed data
            self.logger.info(f"Saving processed data to {output_file}")
            df_transformed.to_csv(output_file, index=False)
            
            self.logger.info("Data pipeline completed successfully")
            
            return {
                'data': df_transformed,
                'transformers': transformers
            }
            
        except Exception as e:
            self.logger.error(f"Data pipeline failed: {str(e)}")
            raise

def run_data_pipeline(
    input_file: Path = RAW_DATA_FILE,
    output_file: Path = FINAL_DATA_FILE,
    **kwargs
) -> Dict[str, Any]:
    """
    Run the data pipeline with default components.
    
    Args:
        input_file: Path to the input data file.
        output_file: Path to save the processed data.
        **kwargs: Additional arguments to pass to the pipeline components.
        
    Returns:
        Dict[str, Any]: Dictionary containing the processed data and transformers.
    """
    pipeline = DataPipeline(**kwargs)
    return pipeline.run(input_file, output_file)

if __name__ == "__main__":
    # Example usage
    import logging.config
    from ..config import LOGGING_CONFIG
    
    # Configure logging
    logging.config.dictConfig(LOGGING_CONFIG)
    
    # Run the pipeline
    result = run_data_pipeline()
    print(f"Pipeline completed. Processed data shape: {result['data'].shape}") 
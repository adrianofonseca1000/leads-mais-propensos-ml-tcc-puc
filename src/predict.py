"""
Prediction module for the Lead Scoring project.
Handles loading trained models and making predictions.
"""

import logging
import logging.config
from pathlib import Path
from typing import Dict, Any, Optional, Union
import pandas as pd
import joblib

from .config import (
    TRAINED_MODELS_DIR,
    LOGGING_CONFIG
)
from .data.loader import DataLoaderFactory
from .data.transformer import DataTransformerFactory

class Predictor:
    """Class for making predictions using trained models."""
    
    def __init__(
        self,
        model_path: Optional[Path] = None,
        model_type: Optional[str] = None
    ):
        """
        Initialize the predictor.
        
        Args:
            model_path: Path to the trained model. If None, uses the default path.
            model_type: Type of model to load. Required if model_path is None.
        """
        self.logger = logging.getLogger(__name__)
        
        # Set model path
        if model_path is None:
            if model_type is None:
                raise ValueError("Either model_path or model_type must be provided")
            model_path = TRAINED_MODELS_DIR / f"{model_type}_model.joblib"
        
        # Load model
        self.logger.info(f"Loading model from {model_path}")
        self.model = joblib.load(model_path)
        
        # Load transformers if available
        transformers_path = model_path.parent / f"{model_path.stem}_transformers.joblib"
        if transformers_path.exists():
            self.logger.info(f"Loading transformers from {transformers_path}")
            self.transformers = joblib.load(transformers_path)
        else:
            self.transformers = None
    
    def predict(
        self,
        data: Union[pd.DataFrame, Path],
        threshold: float = 0.5,
        return_proba: bool = False
    ) -> Union[pd.Series, pd.DataFrame]:
        """
        Make predictions on the given data.
        
        Args:
            data: Input data or path to data file.
            threshold: Probability threshold for binary classification.
            return_proba: Whether to return probabilities instead of class labels.
            
        Returns:
            Union[pd.Series, pd.DataFrame]: Predictions or probabilities.
        """
        # Load data if path is provided
        if isinstance(data, Path):
            self.logger.info(f"Loading data from {data}")
            loader = DataLoaderFactory.create_loader()
            data = loader.load(data)
        
        # Transform data if transformers are available
        if self.transformers is not None:
            self.logger.info("Transforming data")
            transformer = DataTransformerFactory.create_transformer()
            data, _ = transformer.transform(data)
        
        # Make predictions
        self.logger.info("Making predictions")
        if return_proba:
            predictions = pd.DataFrame(
                self.model.predict_proba(data),
                columns=['prob_0', 'prob_1']
            )
        else:
            proba = self.model.predict_proba(data)[:, 1]
            predictions = pd.Series(
                (proba >= threshold).astype(int),
                index=data.index
            )
        
        return predictions
    
    def predict_batch(
        self,
        data_paths: list[Path],
        output_dir: Path,
        threshold: float = 0.5,
        return_proba: bool = False
    ) -> None:
        """
        Make predictions on multiple data files.
        
        Args:
            data_paths: List of paths to data files.
            output_dir: Directory to save predictions.
            threshold: Probability threshold for binary classification.
            return_proba: Whether to return probabilities instead of class labels.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for data_path in data_paths:
            self.logger.info(f"Processing {data_path}")
            
            # Make predictions
            predictions = self.predict(
                data_path,
                threshold=threshold,
                return_proba=return_proba
            )
            
            # Save predictions
            output_path = output_dir / f"{data_path.stem}_predictions.csv"
            predictions.to_csv(output_path)
            self.logger.info(f"Saved predictions to {output_path}")

def predict(
    data: Union[pd.DataFrame, Path],
    model_path: Optional[Path] = None,
    model_type: Optional[str] = None,
    **kwargs
) -> Union[pd.Series, pd.DataFrame]:
    """
    Make predictions using a trained model.
    
    Args:
        data: Input data or path to data file.
        model_path: Path to the trained model.
        model_type: Type of model to load.
        **kwargs: Additional arguments for the predictor.
        
    Returns:
        Union[pd.Series, pd.DataFrame]: Predictions or probabilities.
    """
    predictor = Predictor(model_path=model_path, model_type=model_type)
    return predictor.predict(data, **kwargs)

if __name__ == "__main__":
    # Example usage
    import logging.config
    logging.config.dictConfig(LOGGING_CONFIG)
    
    # Make predictions
    predictions = predict(
        data=Path("data/test.csv"),
        model_type="random_forest",
        return_proba=True
    )
    print("Predictions:")
    print(predictions.head()) 
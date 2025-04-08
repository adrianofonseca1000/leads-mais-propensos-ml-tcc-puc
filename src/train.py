"""
Main training module for the Lead Scoring project.
Handles model training, evaluation, and saving.
"""

import logging
import logging.config
from pathlib import Path
from typing import Dict, Any, Optional
import pandas as pd
from sklearn.model_selection import train_test_split
import joblib

from .config import (
    FINAL_DATA_FILE,
    TRAINED_MODELS_DIR,
    TARGET_COLUMN,
    RANDOM_SEED,
    TEST_SIZE,
    LOGGING_CONFIG
)
from .models import ModelFactory
from .data.loader import DataLoaderFactory

def train_model(
    model_type: str,
    data_path: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Train a model on the given data.
    
    Args:
        model_type: Type of model to train.
        data_path: Path to the training data. If None, uses the default path.
        output_dir: Directory to save the trained model. If None, uses the default directory.
        **kwargs: Additional parameters for the model and training process.
        
    Returns:
        Dict[str, Any]: Dictionary containing the trained model and evaluation results.
    """
    # Configure logging
    logging.config.dictConfig(LOGGING_CONFIG)
    logger = logging.getLogger(__name__)
    
    try:
        # Set default paths if not provided
        data_path = data_path or FINAL_DATA_FILE
        output_dir = output_dir or TRAINED_MODELS_DIR
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load data
        logger.info(f"Loading data from {data_path}")
        loader = DataLoaderFactory.create_loader()
        df = loader.load(data_path)
        
        # Split data
        logger.info("Splitting data into train and test sets")
        X = df.drop(columns=[TARGET_COLUMN])
        y = df[TARGET_COLUMN]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=TEST_SIZE,
            random_state=RANDOM_SEED,
            stratify=y
        )
        
        # Create and train model
        logger.info(f"Creating {model_type} model")
        model = ModelFactory.create_model(model_type, **kwargs)
        
        logger.info("Training model")
        model.fit(X_train, y_train)
        
        # Evaluate model
        logger.info("Evaluating model")
        train_metrics = model.evaluate(X_train, y_train)
        test_metrics = model.evaluate(X_test, y_test)
        
        # Get feature importance
        feature_importance = model.get_feature_importance()
        
        # Save model
        model_path = output_dir / f"{model_type}_model.joblib"
        logger.info(f"Saving model to {model_path}")
        joblib.dump(model, model_path)
        
        # Save feature importance if available
        if feature_importance is not None:
            importance_path = output_dir / f"{model_type}_feature_importance.csv"
            feature_importance.to_csv(importance_path)
            logger.info(f"Saved feature importance to {importance_path}")
        
        # Return results
        results = {
            'model': model,
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
            'feature_importance': feature_importance,
            'model_path': model_path
        }
        
        logger.info("Training completed successfully")
        return results
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise

if __name__ == "__main__":
    # Example usage
    results = train_model('random_forest')
    print("Training results:")
    print(f"Test ROC AUC: {results['test_metrics']['roc_auc']:.4f}")
    print(f"Model saved to: {results['model_path']}") 
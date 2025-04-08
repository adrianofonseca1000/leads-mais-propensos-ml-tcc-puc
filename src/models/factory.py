"""
Model factory module for the Lead Scoring project.
Implements the factory pattern for creating model instances.
"""

from typing import Dict, Any, Type
from .base import BaseModel
from .random_forest import RandomForestModel
from .xgboost import XGBoostModel
from .logistic_regression import LogisticRegressionModel

class ModelFactory:
    """Factory class for creating model instances."""
    
    _model_classes: Dict[str, Type[BaseModel]] = {
        'random_forest': RandomForestModel,
        'xgboost': XGBoostModel,
        'logistic_regression': LogisticRegressionModel
    }
    
    @classmethod
    def create_model(cls, model_type: str, **kwargs) -> BaseModel:
        """
        Create a model instance.
        
        Args:
            model_type: Type of model to create.
            **kwargs: Additional parameters for the model.
            
        Returns:
            BaseModel: Model instance.
            
        Raises:
            ValueError: If the model type is not supported.
        """
        model_type = model_type.lower()
        
        if model_type not in cls._model_classes:
            raise ValueError(
                f"Unsupported model type: {model_type}. "
                f"Supported types are: {list(cls._model_classes.keys())}"
            )
        
        return cls._model_classes[model_type](**kwargs)
    
    @classmethod
    def get_supported_models(cls) -> Dict[str, Type[BaseModel]]:
        """
        Get a dictionary of supported model types and their classes.
        
        Returns:
            Dict[str, Type[BaseModel]]: Dictionary of supported model types and classes.
        """
        return cls._model_classes.copy() 
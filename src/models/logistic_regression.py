"""
Logistic Regression model implementation for the Lead Scoring project.
"""

from typing import Dict, Any
from sklearn.linear_model import LogisticRegression
from .base import BaseModel

class LogisticRegressionModel(BaseModel):
    """Logistic Regression model implementation."""
    
    def __init__(self, **kwargs):
        """
        Initialize the Logistic Regression model.
        
        Args:
            **kwargs: Additional parameters for the Logistic Regression classifier.
        """
        super().__init__(name='Logistic Regression', **kwargs)
    
    def _create_model(self, **kwargs) -> LogisticRegression:
        """
        Create and return a Logistic Regression classifier.
        
        Args:
            **kwargs: Additional parameters for the Logistic Regression classifier.
            
        Returns:
            LogisticRegression: Logistic Regression classifier instance.
        """
        return LogisticRegression(
            random_state=42,
            max_iter=1000,
            n_jobs=-1,
            **kwargs
        )
    
    def get_param_grid(self) -> Dict[str, Any]:
        """
        Return the parameter grid for hyperparameter tuning.
        
        Returns:
            Dict[str, Any]: Parameter grid for Logistic Regression.
        """
        return {
            'C': [0.1, 1.0, 10.0],
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear'],
            'class_weight': [None, 'balanced']
        } 
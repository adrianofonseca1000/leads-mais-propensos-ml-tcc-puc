"""
Random Forest model implementation for the Lead Scoring project.
"""

from typing import Dict, Any
from sklearn.ensemble import RandomForestClassifier
from .base import BaseModel

class RandomForestModel(BaseModel):
    """Random Forest model implementation."""
    
    def __init__(self, **kwargs):
        """
        Initialize the Random Forest model.
        
        Args:
            **kwargs: Additional parameters for the Random Forest classifier.
        """
        super().__init__(name='Random Forest', **kwargs)
    
    def _create_model(self, **kwargs) -> RandomForestClassifier:
        """
        Create and return a Random Forest classifier.
        
        Args:
            **kwargs: Additional parameters for the Random Forest classifier.
            
        Returns:
            RandomForestClassifier: Random Forest classifier instance.
        """
        return RandomForestClassifier(
            random_state=42,
            n_jobs=-1,
            **kwargs
        )
    
    def get_param_grid(self) -> Dict[str, Any]:
        """
        Return the parameter grid for hyperparameter tuning.
        
        Returns:
            Dict[str, Any]: Parameter grid for Random Forest.
        """
        return {
            'n_estimators': [100, 200, 300],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2']
        } 
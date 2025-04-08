"""
XGBoost model implementation for the Lead Scoring project.
"""

from typing import Dict, Any
import xgboost as xgb
from .base import BaseModel

class XGBoostModel(BaseModel):
    """XGBoost model implementation."""
    
    def __init__(self, **kwargs):
        """
        Initialize the XGBoost model.
        
        Args:
            **kwargs: Additional parameters for the XGBoost classifier.
        """
        super().__init__(name='XGBoost', **kwargs)
    
    def _create_model(self, **kwargs) -> xgb.XGBClassifier:
        """
        Create and return an XGBoost classifier.
        
        Args:
            **kwargs: Additional parameters for the XGBoost classifier.
            
        Returns:
            xgb.XGBClassifier: XGBoost classifier instance.
        """
        return xgb.XGBClassifier(
            random_state=42,
            n_jobs=-1,
            **kwargs
        )
    
    def get_param_grid(self) -> Dict[str, Any]:
        """
        Return the parameter grid for hyperparameter tuning.
        
        Returns:
            Dict[str, Any]: Parameter grid for XGBoost.
        """
        return {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0],
            'min_child_weight': [1, 3, 5]
        } 
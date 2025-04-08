"""
Base model interface for the Lead Scoring project.
Defines the contract that all model implementations must follow.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import logging

class BaseModel(ABC):
    """Abstract base class for all machine learning models."""
    
    def __init__(self, name: str, **kwargs):
        """
        Initialize the base model.
        
        Args:
            name: Name of the model.
            **kwargs: Additional model-specific parameters.
        """
        self.name = name
        self.logger = logging.getLogger(__name__)
        self.model = None
        self.best_params_ = None
        self.feature_importance_ = None
    
    @abstractmethod
    def _create_model(self, **kwargs) -> Any:
        """Create and return the model instance."""
        pass
    
    @abstractmethod
    def get_param_grid(self) -> Dict[str, Any]:
        """Return the parameter grid for hyperparameter tuning."""
        pass
    
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        cv: int = 5,
        n_iter: int = 10,
        random_state: int = 42,
        search_type: str = 'grid'
    ) -> None:
        """
        Fit the model with hyperparameter tuning.
        
        Args:
            X: Feature matrix.
            y: Target vector.
            cv: Number of cross-validation folds.
            n_iter: Number of iterations for randomized search.
            random_state: Random seed for reproducibility.
            search_type: Type of search ('grid' or 'random').
        """
        self.logger.info(f"Fitting {self.name} model")
        
        # Create base model
        self.model = self._create_model()
        
        # Get parameter grid
        param_grid = self.get_param_grid()
        
        # Perform hyperparameter search
        if search_type == 'grid':
            search = GridSearchCV(
                self.model,
                param_grid,
                cv=cv,
                scoring='roc_auc',
                n_jobs=-1
            )
        else:
            search = RandomizedSearchCV(
                self.model,
                param_grid,
                cv=cv,
                n_iter=n_iter,
                scoring='roc_auc',
                n_jobs=-1,
                random_state=random_state
            )
        
        # Fit the search
        search.fit(X, y)
        
        # Store results
        self.model = search.best_estimator_
        self.best_params_ = search.best_params_
        
        self.logger.info(f"Best parameters: {self.best_params_}")
        self.logger.info(f"Best score: {search.best_score_:.4f}")
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using the fitted model.
        
        Args:
            X: Feature matrix.
            
        Returns:
            np.ndarray: Predicted class labels.
        """
        if self.model is None:
            raise ValueError("Model has not been fitted yet")
        
        return self.model.predict(X)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make probability predictions using the fitted model.
        
        Args:
            X: Feature matrix.
            
        Returns:
            np.ndarray: Predicted probabilities.
        """
        if self.model is None:
            raise ValueError("Model has not been fitted yet")
        
        return self.model.predict_proba(X)
    
    def evaluate(
        self,
        X: pd.DataFrame,
        y: pd.Series
    ) -> Dict[str, float]:
        """
        Evaluate the model on the given data.
        
        Args:
            X: Feature matrix.
            y: True labels.
            
        Returns:
            Dict[str, float]: Dictionary of evaluation metrics.
        """
        if self.model is None:
            raise ValueError("Model has not been fitted yet")
        
        # Make predictions
        y_pred = self.predict(X)
        y_pred_proba = self.predict_proba(X)[:, 1]
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred),
            'recall': recall_score(y, y_pred),
            'f1': f1_score(y, y_pred),
            'roc_auc': roc_auc_score(y, y_pred_proba)
        }
        
        self.logger.info("Model evaluation results:")
        for metric, value in metrics.items():
            self.logger.info(f"{metric}: {value:.4f}")
        
        return metrics
    
    def get_feature_importance(self) -> Optional[pd.Series]:
        """
        Get feature importance scores if available.
        
        Returns:
            Optional[pd.Series]: Feature importance scores if available, None otherwise.
        """
        if self.model is None:
            raise ValueError("Model has not been fitted yet")
        
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importance_ = pd.Series(
                self.model.feature_importances_,
                index=self.model.feature_names_in_
            ).sort_values(ascending=False)
        
        return self.feature_importance_ 
"""
Base class for Learning with Label Proportions algorithms.
"""
from abc import ABC, abstractmethod
import numpy as np


class BaseLLPClassifier(ABC):
    """
    Abstract base class for Learning with Label Proportions classifiers.
    
    All LLP classifiers should inherit from this class and implement
    the fit and predict methods.
    """
    
    def __init__(self):
        """Initialize the base classifier."""
        self.is_fitted_ = False
    
    @abstractmethod
    def fit(self, X_bags, proportions):
        """
        Fit the classifier on bags with known label proportions.
        
        Parameters
        ----------
        X_bags : list of arrays
            List of K bags, where each bag is an array of shape (n_samples_k, n_features)
        proportions : array-like of shape (K,)
            Proportion of positive class in each bag (for binary classification)
            
        Returns
        -------
        self : object
            Returns self.
        """
        pass
    
    @abstractmethod
    def predict(self, X):
        """
        Predict class labels for samples in X.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict
            
        Returns
        -------
        y_pred : array-like of shape (n_samples,)
            Predicted class labels
        """
        pass
    
    def score(self, X, y_true):
        """
        Calculate accuracy score.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples
        y_true : array-like of shape (n_samples,)
            True labels
            
        Returns
        -------
        score : float
            Accuracy score
        """
        y_pred = self.predict(X)
        return np.mean(y_pred == y_true)

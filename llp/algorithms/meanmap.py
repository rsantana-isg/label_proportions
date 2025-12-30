"""
MeanMap algorithm for Learning with Label Proportions.

Based on the paper:
"Estimating Labels from Label Proportions" by Quadrianto et al. (2008)
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from llp.algorithms.base import BaseLLPClassifier


class MeanMap(BaseLLPClassifier):
    """
    MeanMap algorithm for learning from label proportions.
    
    This method estimates the mean feature vectors for each class using
    the bag means and label proportions. It then uses these estimates
    to train a classifier.
    
    Parameters
    ----------
    C : float, default=1.0
        Regularization parameter for the logistic regression
    max_iter : int, default=1000
        Maximum number of iterations for logistic regression
    random_state : int or None, default=None
        Random seed for reproducibility
    """
    
    def __init__(self, C=1.0, max_iter=1000, random_state=None):
        super().__init__()
        self.C = C
        self.max_iter = max_iter
        self.random_state = random_state
        self.classifier = None
        self.mu_pos = None
        self.mu_neg = None
    
    def fit(self, X_bags, proportions):
        """
        Fit the MeanMap classifier.
        
        Parameters
        ----------
        X_bags : list of arrays
            List of K bags, where each bag is an array of shape (n_samples_k, n_features)
        proportions : array-like of shape (K,)
            Proportion of positive class in each bag
            
        Returns
        -------
        self : object
            Returns self.
        """
        proportions = np.array(proportions)
        K = len(X_bags)
        
        # Compute bag means
        bag_means = np.array([np.mean(bag, axis=0) for bag in X_bags])
        
        # Estimate class means using the MeanMap approach
        # The bag mean is: mu_bag_k = p_k * mu_pos + (1 - p_k) * mu_neg
        # This gives us a system of linear equations
        
        # Build the system: A * [mu_pos; mu_neg] = bag_means
        # where A is a K x 2 matrix with A[k, :] = [p_k, 1-p_k]
        
        # We solve this as a least squares problem
        # mu_pos and mu_neg are the mean vectors for positive and negative classes
        
        # Stack proportions to create coefficient matrix
        A = np.column_stack([proportions, 1 - proportions])
        
        # Use least squares to estimate class means
        # We need to solve for each feature dimension separately
        n_features = bag_means.shape[1]
        mu_pos = np.zeros(n_features)
        mu_neg = np.zeros(n_features)
        
        for d in range(n_features):
            # Solve for dimension d
            solution, _, _, _ = np.linalg.lstsq(A, bag_means[:, d], rcond=None)
            mu_pos[d] = solution[0]
            mu_neg[d] = solution[1]
        
        self.mu_pos = mu_pos
        self.mu_neg = mu_neg
        
        # Create training data from estimated means
        # We create synthetic samples around the estimated means
        n_synthetic = 100
        X_synthetic = []
        y_synthetic = []
        
        # Generate synthetic samples for both classes
        # Estimate covariance from all bags
        all_samples = np.vstack(X_bags)
        cov = np.cov(all_samples.T)
        cov_reg = cov + np.eye(n_features) * 0.01  # Add small regularization
        
        for _ in range(n_synthetic // 2):
            X_synthetic.append(np.random.multivariate_normal(mu_pos, cov_reg))
            y_synthetic.append(1)
            X_synthetic.append(np.random.multivariate_normal(mu_neg, cov_reg))
            y_synthetic.append(0)
        
        X_synthetic = np.array(X_synthetic)
        y_synthetic = np.array(y_synthetic)
        
        # Train logistic regression on synthetic data
        self.classifier = LogisticRegression(
            C=self.C,
            max_iter=self.max_iter,
            random_state=self.random_state
        )
        self.classifier.fit(X_synthetic, y_synthetic)
        
        self.is_fitted_ = True
        return self
    
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
        if not self.is_fitted_:
            raise ValueError("The classifier has not been fitted yet.")
        
        return self.classifier.predict(X)
    
    def predict_proba(self, X):
        """
        Predict class probabilities for samples in X.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict
            
        Returns
        -------
        proba : array-like of shape (n_samples, 2)
            Class probabilities
        """
        if not self.is_fitted_:
            raise ValueError("The classifier has not been fitted yet.")
        
        return self.classifier.predict_proba(X)

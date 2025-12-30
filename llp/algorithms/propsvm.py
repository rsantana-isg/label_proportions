"""
Proportion SVM (∝SVM) algorithm for Learning with Label Proportions.

Based on the paper:
"∝SVM for Learning with Label Proportions" by Yu et al. (2013)
"""
import numpy as np
from sklearn.svm import SVC
from llp.algorithms.base import BaseLLPClassifier


class PropSVM(BaseLLPClassifier):
    """
    Proportion SVM (∝SVM) for learning from label proportions.
    
    This method uses an alternating optimization approach:
    1. Fix labels, optimize SVM parameters
    2. Fix SVM parameters, optimize labels (respecting bag proportions)
    
    Parameters
    ----------
    C : float, default=1.0
        Regularization parameter for the SVM
    kernel : str, default='rbf'
        Kernel type for SVM ('linear', 'rbf', 'poly', 'sigmoid')
    max_iter : int, default=100
        Maximum number of alternating optimization iterations
    tol : float, default=1e-4
        Convergence tolerance
    random_state : int or None, default=None
        Random seed for reproducibility
    """
    
    def __init__(self, C=1.0, kernel='rbf', max_iter=100, tol=1e-4, random_state=None):
        super().__init__()
        self.C = C
        self.kernel = kernel
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.svm = None
        self.labels_ = None
    
    def _initialize_labels(self, X_bags, proportions):
        """
        Initialize instance labels randomly while respecting bag proportions.
        
        Parameters
        ----------
        X_bags : list of arrays
            List of bags
        proportions : array
            Target proportions for each bag
            
        Returns
        -------
        labels : list of arrays
            Initial labels for each bag
        """
        labels = []
        
        for bag, prop in zip(X_bags, proportions):
            n_samples = len(bag)
            n_positive = int(np.round(prop * n_samples))
            
            # Create labels with correct proportion
            bag_labels = np.zeros(n_samples)
            if n_positive > 0:
                positive_indices = np.random.choice(n_samples, n_positive, replace=False)
                bag_labels[positive_indices] = 1
            
            labels.append(bag_labels)
        
        return labels
    
    def _optimize_labels(self, X_bags, proportions, svm):
        """
        Optimize instance labels given fixed SVM, respecting bag proportions.
        
        Parameters
        ----------
        X_bags : list of arrays
            List of bags
        proportions : array
            Target proportions for each bag
        svm : SVC
            Trained SVM classifier
            
        Returns
        -------
        labels : list of arrays
            Optimized labels for each bag
        """
        labels = []
        
        for bag, prop in zip(X_bags, proportions):
            n_samples = len(bag)
            n_positive = int(np.round(prop * n_samples))
            
            # Get decision function values (distance from hyperplane)
            decision_values = svm.decision_function(bag)
            
            # Assign positive labels to samples with highest decision values
            if n_positive > 0:
                top_indices = np.argsort(decision_values)[-n_positive:]
                bag_labels = np.zeros(n_samples)
                bag_labels[top_indices] = 1
            else:
                bag_labels = np.zeros(n_samples)
            
            labels.append(bag_labels)
        
        return labels
    
    def fit(self, X_bags, proportions):
        """
        Fit the PropSVM classifier using alternating optimization.
        
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
        
        # Initialize labels randomly
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        labels = self._initialize_labels(X_bags, proportions)
        
        # Flatten bags and labels for training
        X_flat = np.vstack(X_bags)
        y_flat = np.concatenate(labels)
        
        prev_accuracy = 0
        
        for iteration in range(self.max_iter):
            # Step 1: Train SVM with current labels
            self.svm = SVC(
                C=self.C,
                kernel=self.kernel,
                random_state=self.random_state
            )
            self.svm.fit(X_flat, y_flat)
            
            # Step 2: Update labels based on SVM predictions
            new_labels = self._optimize_labels(X_bags, proportions, self.svm)
            new_y_flat = np.concatenate(new_labels)
            
            # Check convergence
            accuracy = np.mean(y_flat == new_y_flat)
            
            if abs(accuracy - prev_accuracy) < self.tol:
                break
            
            labels = new_labels
            y_flat = new_y_flat
            prev_accuracy = accuracy
        
        # Store final labels
        self.labels_ = labels
        
        # Train final SVM
        self.svm = SVC(
            C=self.C,
            kernel=self.kernel,
            probability=True,
            random_state=self.random_state
        )
        self.svm.fit(X_flat, y_flat)
        
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
        
        return self.svm.predict(X)
    
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
        
        return self.svm.predict_proba(X)

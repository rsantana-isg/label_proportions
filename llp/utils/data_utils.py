"""
Utility functions for data generation and manipulation in LLP.
"""
import numpy as np
from sklearn.datasets import make_classification


def create_bags(X, y, bag_sizes, shuffle=True):
    """
    Create bags from labeled data.
    
    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Feature matrix
    y : array-like of shape (n_samples,)
        Labels (binary: 0 or 1)
    bag_sizes : list of int
        Size of each bag
    shuffle : bool, default=True
        Whether to shuffle data before creating bags
        
    Returns
    -------
    X_bags : list of arrays
        List of bags, each containing feature vectors
    proportions : array
        Proportion of positive class (label=1) in each bag
    bag_indices : list of arrays
        Indices of samples in each bag (for tracking)
    """
    n_samples = len(X)
    
    if shuffle:
        indices = np.random.permutation(n_samples)
    else:
        indices = np.arange(n_samples)
    
    X_shuffled = X[indices]
    y_shuffled = y[indices]
    
    X_bags = []
    proportions = []
    bag_indices = []
    
    start_idx = 0
    for bag_size in bag_sizes:
        end_idx = start_idx + bag_size
        if end_idx > n_samples:
            end_idx = n_samples
            
        X_bag = X_shuffled[start_idx:end_idx]
        y_bag = y_shuffled[start_idx:end_idx]
        
        X_bags.append(X_bag)
        proportions.append(np.mean(y_bag))
        bag_indices.append(indices[start_idx:end_idx])
        
        start_idx = end_idx
        
        if start_idx >= n_samples:
            break
    
    return X_bags, np.array(proportions), bag_indices


def generate_llp_dataset(n_samples=1000, n_features=20, n_bags=10, 
                         bag_size_range=(50, 150), random_state=None):
    """
    Generate a synthetic dataset for LLP with known ground truth.
    
    Parameters
    ----------
    n_samples : int, default=1000
        Total number of samples
    n_features : int, default=20
        Number of features
    n_bags : int, default=10
        Number of bags to create
    bag_size_range : tuple of int, default=(50, 150)
        Range for bag sizes (min, max)
    random_state : int or None, default=None
        Random seed for reproducibility
        
    Returns
    -------
    X_bags : list of arrays
        Training bags
    proportions : array
        Label proportions for training bags
    X_test : array
        Test samples
    y_test : array
        Test labels
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    # Generate binary classification dataset
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=int(n_features * 0.7),
        n_redundant=int(n_features * 0.2),
        n_clusters_per_class=2,
        flip_y=0.01,
        class_sep=1.0,
        random_state=random_state
    )
    
    # Split into train and test
    test_size = int(0.3 * n_samples)
    train_size = n_samples - test_size
    
    indices = np.random.permutation(n_samples)
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]
    
    X_train = X[train_indices]
    y_train = y[train_indices]
    X_test = X[test_indices]
    y_test = y[test_indices]
    
    # Create bags from training data
    bag_sizes = np.random.randint(bag_size_range[0], bag_size_range[1], size=n_bags)
    # Adjust bag sizes to fit training data
    total_needed = np.sum(bag_sizes)
    if total_needed > train_size:
        bag_sizes = (bag_sizes * train_size / total_needed).astype(int)
        bag_sizes[-1] = train_size - np.sum(bag_sizes[:-1])
    
    X_bags, proportions, _ = create_bags(X_train, y_train, bag_sizes, shuffle=True)
    
    return X_bags, proportions, X_test, y_test

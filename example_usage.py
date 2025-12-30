"""
Example script demonstrating the usage of LLP algorithms.

This script shows how to:
1. Generate synthetic LLP datasets
2. Train different LLP algorithms
3. Evaluate and compare their performance
"""
import numpy as np
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score, confusion_matrix

from llp.algorithms.meanmap import MeanMap
from llp.algorithms.propsvm import PropSVM
from llp.utils.data_utils import create_bags, generate_llp_dataset


def example_1_basic_usage():
    """Example 1: Basic usage with generated dataset."""
    print("=" * 80)
    print("Example 1: Basic Usage")
    print("=" * 80)
    
    # Generate dataset
    X_bags, proportions, X_test, y_test = generate_llp_dataset(
        n_samples=500,
        n_features=10,
        n_bags=5,
        random_state=123
    )
    
    print(f"\nDataset information:")
    print(f"  Number of bags: {len(X_bags)}")
    print(f"  Bag sizes: {[len(bag) for bag in X_bags]}")
    print(f"  Label proportions: {proportions.round(3)}")
    print(f"  Test set size: {len(X_test)}")
    
    # Train MeanMap
    print("\nTraining MeanMap...")
    meanmap = MeanMap(random_state=123)
    meanmap.fit(X_bags, proportions)
    y_pred = meanmap.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"  MeanMap Accuracy: {accuracy:.4f}")
    
    # Train PropSVM
    print("\nTraining PropSVM...")
    propsvm = PropSVM(kernel='rbf', max_iter=30, random_state=123)
    propsvm.fit(X_bags, proportions)
    y_pred = propsvm.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"  PropSVM Accuracy: {accuracy:.4f}")


def example_2_custom_dataset():
    """Example 2: Using custom dataset."""
    print("\n" + "=" * 80)
    print("Example 2: Custom Dataset")
    print("=" * 80)
    
    # Create custom dataset
    X, y = make_classification(
        n_samples=400,
        n_features=15,
        n_informative=10,
        n_redundant=3,
        n_clusters_per_class=2,
        random_state=456
    )
    
    print(f"\nDataset created:")
    print(f"  Total samples: {len(X)}")
    print(f"  Features: {X.shape[1]}")
    print(f"  Class distribution: {np.bincount(y)}")
    
    # Split into train and test
    split_idx = int(0.7 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Create bags from training data
    bag_sizes = [70, 70, 70, 70]
    X_bags, proportions, _ = create_bags(X_train, y_train, bag_sizes)
    
    print(f"\nBags created:")
    print(f"  Number of bags: {len(X_bags)}")
    print(f"  Label proportions: {proportions.round(3)}")
    
    # Train and evaluate
    for name, model in [
        ('MeanMap', MeanMap(random_state=456)),
        ('PropSVM', PropSVM(kernel='linear', max_iter=30, random_state=456))
    ]:
        print(f"\n{name}:")
        model.fit(X_bags, proportions)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Confusion Matrix:\n{cm}")


def example_3_different_bag_sizes():
    """Example 3: Testing with different bag sizes."""
    print("\n" + "=" * 80)
    print("Example 3: Impact of Bag Size")
    print("=" * 80)
    
    n_samples = 800
    n_features = 15
    
    print(f"\nTesting with different number of bags:")
    print(f"  Total samples: {n_samples}")
    print(f"  Features: {n_features}")
    
    for n_bags in [5, 10, 20]:
        print(f"\n  Number of bags: {n_bags}")
        
        # Generate dataset
        X_bags, proportions, X_test, y_test = generate_llp_dataset(
            n_samples=n_samples,
            n_features=n_features,
            n_bags=n_bags,
            random_state=789
        )
        
        # Train MeanMap
        meanmap = MeanMap(random_state=789)
        meanmap.fit(X_bags, proportions)
        acc_mm = accuracy_score(y_test, meanmap.predict(X_test))
        
        # Train PropSVM
        propsvm = PropSVM(max_iter=20, random_state=789)
        propsvm.fit(X_bags, proportions)
        acc_ps = accuracy_score(y_test, propsvm.predict(X_test))
        
        print(f"    MeanMap Accuracy: {acc_mm:.4f}")
        print(f"    PropSVM Accuracy: {acc_ps:.4f}")


def example_4_label_proportion_analysis():
    """Example 4: Analyzing learned vs actual proportions."""
    print("\n" + "=" * 80)
    print("Example 4: Label Proportion Analysis")
    print("=" * 80)
    
    # Generate dataset
    X_bags, true_proportions, X_test, y_test = generate_llp_dataset(
        n_samples=600,
        n_features=10,
        n_bags=6,
        random_state=999
    )
    
    print(f"\nTrue label proportions in training bags:")
    for i, prop in enumerate(true_proportions):
        print(f"  Bag {i+1}: {prop:.3f}")
    
    # Train model
    model = PropSVM(max_iter=30, random_state=999)
    model.fit(X_bags, true_proportions)
    
    # Predict on training bags to see learned proportions
    print(f"\nLearned label proportions (predictions on training bags):")
    for i, bag in enumerate(X_bags):
        predictions = model.predict(bag)
        learned_prop = np.mean(predictions)
        print(f"  Bag {i+1}: {learned_prop:.3f} (true: {true_proportions[i]:.3f})")
    
    # Test accuracy
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nTest set accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    np.random.seed(42)
    
    # Run all examples
    example_1_basic_usage()
    example_2_custom_dataset()
    example_3_different_bag_sizes()
    example_4_label_proportion_analysis()
    
    print("\n" + "=" * 80)
    print("All examples completed successfully!")
    print("=" * 80)

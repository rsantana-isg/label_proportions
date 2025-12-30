"""
Comparison script for LLP algorithms.

This script compares different Learning with Label Proportions algorithms
in terms of classification accuracy and computational time.
"""
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report

from llp.algorithms.propsvm import PropSVM
from llp.algorithms.meanmap import MeanMap
from llp.utils.data_utils import generate_llp_dataset


def run_comparison(n_samples=1000, n_features=20, n_bags=10, random_state=42):
    """
    Run comparison of LLP algorithms.
    
    Parameters
    ----------
    n_samples : int
        Total number of samples
    n_features : int
        Number of features
    n_bags : int
        Number of bags
    random_state : int
        Random seed
        
    Returns
    -------
    results : dict
        Dictionary containing results for each algorithm
    """
    print("=" * 80)
    print("Learning with Label Proportions - Algorithm Comparison")
    print("=" * 80)
    print(f"\nDataset Configuration:")
    print(f"  - Total samples: {n_samples}")
    print(f"  - Features: {n_features}")
    print(f"  - Number of bags: {n_bags}")
    print(f"  - Random state: {random_state}")
    
    # Generate dataset
    print("\nGenerating synthetic dataset...")
    X_bags, proportions, X_test, y_test = generate_llp_dataset(
        n_samples=n_samples,
        n_features=n_features,
        n_bags=n_bags,
        bag_size_range=(50, 100),
        random_state=random_state
    )
    
    print(f"  - Training bags: {len(X_bags)}")
    print(f"  - Test samples: {len(X_test)}")
    print(f"  - Label proportions: {proportions}")
    
    # Initialize algorithms
    algorithms = {
        'MeanMap': MeanMap(C=1.0, random_state=random_state),
        'PropSVM': PropSVM(C=1.0, kernel='rbf', max_iter=50, random_state=random_state),
    }
    
    results = {}
    
    print("\n" + "=" * 80)
    print("Training and Evaluation")
    print("=" * 80)
    
    for name, algorithm in algorithms.items():
        print(f"\n{name}:")
        print("-" * 40)
        
        # Training
        print("  Training...")
        start_time = time.time()
        algorithm.fit(X_bags, proportions)
        training_time = time.time() - start_time
        print(f"    Training time: {training_time:.4f} seconds")
        
        # Prediction
        print("  Predicting...")
        start_time = time.time()
        y_pred = algorithm.predict(X_test)
        prediction_time = time.time() - start_time
        print(f"    Prediction time: {prediction_time:.4f} seconds")
        
        # Evaluation
        accuracy = accuracy_score(y_test, y_pred)
        print(f"    Accuracy: {accuracy:.4f}")
        
        results[name] = {
            'accuracy': accuracy,
            'training_time': training_time,
            'prediction_time': prediction_time,
            'total_time': training_time + prediction_time,
            'y_pred': y_pred
        }
        
        # Classification report
        print("\n  Classification Report:")
        report = classification_report(y_test, y_pred, target_names=['Class 0', 'Class 1'])
        for line in report.split('\n'):
            print(f"    {line}")
    
    return results, X_test, y_test


def plot_results(results):
    """
    Plot comparison results.
    
    Parameters
    ----------
    results : dict
        Results dictionary from run_comparison
    """
    algorithms = list(results.keys())
    accuracies = [results[alg]['accuracy'] for alg in algorithms]
    training_times = [results[alg]['training_time'] for alg in algorithms]
    total_times = [results[alg]['total_time'] for alg in algorithms]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Accuracy comparison
    axes[0].bar(algorithms, accuracies, color=['#1f77b4', '#ff7f0e'])
    axes[0].set_ylabel('Accuracy')
    axes[0].set_title('Classification Accuracy')
    axes[0].set_ylim([0, 1])
    axes[0].grid(axis='y', alpha=0.3)
    
    # Training time comparison
    axes[1].bar(algorithms, training_times, color=['#1f77b4', '#ff7f0e'])
    axes[1].set_ylabel('Time (seconds)')
    axes[1].set_title('Training Time')
    axes[1].grid(axis='y', alpha=0.3)
    
    # Total time comparison
    axes[2].bar(algorithms, total_times, color=['#1f77b4', '#ff7f0e'])
    axes[2].set_ylabel('Time (seconds)')
    axes[2].set_title('Total Time (Training + Prediction)')
    axes[2].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('comparison_results.png', dpi=300, bbox_inches='tight')
    print("\nPlot saved as 'comparison_results.png'")
    

def print_summary(results):
    """
    Print summary table of results.
    
    Parameters
    ----------
    results : dict
        Results dictionary from run_comparison
    """
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"\n{'Algorithm':<15} {'Accuracy':<12} {'Train Time':<15} {'Total Time':<15}")
    print("-" * 80)
    
    for name, result in results.items():
        print(f"{name:<15} {result['accuracy']:<12.4f} "
              f"{result['training_time']:<15.4f} {result['total_time']:<15.4f}")
    
    # Find best algorithm
    best_accuracy = max(results.items(), key=lambda x: x[1]['accuracy'])
    fastest = min(results.items(), key=lambda x: x[1]['total_time'])
    
    print("\n" + "-" * 80)
    print(f"Best Accuracy: {best_accuracy[0]} ({best_accuracy[1]['accuracy']:.4f})")
    print(f"Fastest: {fastest[0]} ({fastest[1]['total_time']:.4f} seconds)")


if __name__ == "__main__":
    # Run comparison
    results, X_test, y_test = run_comparison(
        n_samples=1000,
        n_features=20,
        n_bags=10,
        random_state=42
    )
    
    # Print summary
    print_summary(results)
    
    # Plot results
    plot_results(results)
    
    print("\n" + "=" * 80)
    print("Comparison completed successfully!")
    print("=" * 80)

# Learning with Label Proportions

This project implements different approaches to learning classifiers from label proportions (LLP). In the LLP setting, training data is organized into bags, and only the proportion of each class in each bag is known - individual instance labels are not available. The goal is to learn a classifier that can predict labels for individual instances.

## Problem Description

**Learning with Label Proportions (LLP)** is a weakly supervised learning problem where:
- Training instances are provided in groups (bags)
- Only the proportion of each class in each bag is known
- The task is to learn a model that predicts labels of individual instances

This setting arises in many real-world scenarios:
- **Privacy-preserving learning**: Individual labels may be sensitive, but aggregate statistics can be shared (e.g., disease proportions by ZIP code)
- **E-commerce**: Identifying potential customers from aggregated purchase behavior
- **Spam filtering**: Using datasets with mixed spam/non-spam emails with known proportions
- **Voting behavior analysis**: Learning from aggregated voting results across demographic regions

## Implemented Algorithms

### 1. MeanMap

**Reference**: Quadrianto et al., "Estimating Labels from Label Proportions" (ICML 2008)

**Approach**:
- Estimates the mean feature vector for each class using the bag means and label proportions
- Solves a system of linear equations: `bag_mean = proportion * mean_positive + (1 - proportion) * mean_negative`
- Uses estimated class means to train a classifier (logistic regression)

**Advantages**:
- Theoretically sound with consistency guarantees
- Computationally efficient
- Works well when class distributions are similar across bags

**Limitations**:
- Assumes class-conditional distributions are independent of bags
- May not work well when this assumption is violated

### 2. ∝SVM (Proportion SVM)

**Reference**: Yu et al., "∝SVM for Learning with Label Proportions" (ICML 2013)

**Approach**:
- Uses a large-margin framework that explicitly models unknown instance labels
- Alternating optimization:
  1. Fix labels, optimize SVM parameters
  2. Fix SVM, optimize labels (respecting bag proportions)
- Avoids restrictive assumptions about data distribution

**Advantages**:
- More flexible than MeanMap
- Better performance with larger bag sizes
- Explicitly optimizes for classification margin

**Limitations**:
- Computationally more expensive
- Non-convex optimization (may converge to local optima)

## Project Structure

```
label_proportions/
├── llp/                          # Main package
│   ├── __init__.py
│   ├── algorithms/               # Algorithm implementations
│   │   ├── __init__.py
│   │   ├── base.py              # Base class for LLP algorithms
│   │   ├── meanmap.py           # MeanMap implementation
│   │   └── propsvm.py           # ∝SVM implementation
│   └── utils/                    # Utility functions
│       ├── __init__.py
│       └── data_utils.py        # Data generation and manipulation
├── compare_algorithms.py         # Comparison script
├── example_usage.py              # Example usage demonstrations
├── requirements.txt             # Python dependencies
├── README.md                    # This file
└── docs/                        # Research papers
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/rsantana-isg/label_proportions.git
cd label_proportions
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Quick Start Examples

**1. Run the comparison script** to see algorithm performance:

```bash
python compare_algorithms.py
```

This will:
- Generate a synthetic binary classification dataset
- Create bags with known label proportions
- Train both MeanMap and ∝SVM algorithms
- Evaluate accuracy on a held-out test set
- Compare computational time
- Generate visualization plots

**2. Run the example usage script** for more detailed examples:

```bash
python example_usage.py
```

This demonstrates:
- Basic usage with different configurations
- Custom dataset creation
- Impact of bag size on performance
- Label proportion analysis

### Using the Algorithms Directly

```python
import numpy as np
from llp.algorithms.meanmap import MeanMap
from llp.algorithms.propsvm import PropSVM
from llp.utils.data_utils import generate_llp_dataset

# Generate synthetic dataset
X_bags, proportions, X_test, y_test = generate_llp_dataset(
    n_samples=1000,
    n_features=20,
    n_bags=10,
    random_state=42
)

# Train MeanMap
meanmap = MeanMap(C=1.0, random_state=42)
meanmap.fit(X_bags, proportions)
y_pred_mm = meanmap.predict(X_test)
accuracy_mm = np.mean(y_pred_mm == y_test)
print(f"MeanMap Accuracy: {accuracy_mm:.4f}")

# Train PropSVM
propsvm = PropSVM(C=1.0, kernel='rbf', max_iter=50, random_state=42)
propsvm.fit(X_bags, proportions)
y_pred_ps = propsvm.predict(X_test)
accuracy_ps = np.mean(y_pred_ps == y_test)
print(f"PropSVM Accuracy: {accuracy_ps:.4f}")
```

### Creating Custom Datasets

```python
from sklearn.datasets import make_classification
from llp.utils.data_utils import create_bags

# Generate labeled data
X, y = make_classification(n_samples=500, n_features=20, random_state=42)

# Create bags
bag_sizes = [100, 150, 100, 150]
X_bags, proportions, bag_indices = create_bags(X, y, bag_sizes)

# Train an LLP algorithm
from llp.algorithms.meanmap import MeanMap
model = MeanMap()
model.fit(X_bags, proportions)
```

## API Reference

### BaseLLPClassifier

Base class for all LLP algorithms.

**Methods**:
- `fit(X_bags, proportions)`: Fit the classifier on bags with known label proportions
- `predict(X)`: Predict class labels for samples
- `score(X, y_true)`: Calculate accuracy score

### MeanMap

**Parameters**:
- `C` (float, default=1.0): Regularization parameter
- `max_iter` (int, default=1000): Maximum iterations for logistic regression
- `random_state` (int, optional): Random seed

### PropSVM

**Parameters**:
- `C` (float, default=1.0): SVM regularization parameter
- `kernel` (str, default='rbf'): Kernel type ('linear', 'rbf', 'poly', 'sigmoid')
- `max_iter` (int, default=100): Maximum alternating optimization iterations
- `tol` (float, default=1e-4): Convergence tolerance
- `random_state` (int, optional): Random seed

## Dataset Generation

The `generate_llp_dataset` function creates synthetic datasets for testing:

**Parameters**:
- `n_samples` (int): Total number of samples
- `n_features` (int): Number of features
- `n_bags` (int): Number of bags to create
- `bag_size_range` (tuple): Range for bag sizes (min, max)
- `random_state` (int, optional): Random seed

**Returns**:
- `X_bags`: List of training bags
- `proportions`: Label proportions for each bag
- `X_test`: Test samples
- `y_test`: Test labels

## References

1. Novi Quadrianto, Alex J. Smola, Tiberio S. Caetano, and Quoc V. Le. "Estimating Labels from Label Proportions." In *Proceedings of the 25th International Conference on Machine Learning (ICML)*, 2008.

2. Felix X. Yu, Dong Liu, Sanjiv Kumar, Tony Jebara, and Shih-Fu Chang. "∝SVM for Learning with Label Proportions." In *Proceedings of the 30th International Conference on Machine Learning (ICML)*, 2013.

3. Zhiquan Qi, Bo Wang, Fan Meng, and Lingfeng Niu. "Learning With Label Proportions via NPSVM." *IEEE Transactions on Cybernetics*, 2017.

## License

This project is for educational and research purposes. Please refer to the original papers for the theoretical foundations of the algorithms.

## Contributing

Contributions are welcome! Potential areas for improvement:
- Implementation of additional LLP algorithms (e.g., Inverse Calibration, NPSVM)
- Support for multi-class classification
- More sophisticated initialization strategies
- Real-world dataset examples
- Performance optimizations

## Contact

For questions or issues, please open an issue on the GitHub repository.

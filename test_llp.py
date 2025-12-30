"""
Simple tests for LLP algorithms to verify correctness.
"""
import numpy as np
from llp.algorithms.meanmap import MeanMap
from llp.algorithms.propsvm import PropSVM
from llp.utils.data_utils import generate_llp_dataset, create_bags


def test_generate_dataset():
    """Test dataset generation."""
    print("Testing dataset generation...")
    X_bags, proportions, X_test, y_test = generate_llp_dataset(
        n_samples=200, n_features=10, n_bags=5, random_state=42
    )
    
    assert len(X_bags) == 5, "Should have 5 bags"
    assert len(proportions) == 5, "Should have 5 proportions"
    assert all(0 <= p <= 1 for p in proportions), "Proportions should be in [0, 1]"
    assert len(X_test) > 0, "Should have test samples"
    assert len(y_test) == len(X_test), "Test labels should match test samples"
    print("  ✓ Dataset generation works correctly")


def test_create_bags():
    """Test bag creation."""
    print("Testing bag creation...")
    X = np.random.randn(100, 5)
    y = np.random.randint(0, 2, 100)
    bag_sizes = [30, 30, 40]
    
    X_bags, proportions, indices = create_bags(X, y, bag_sizes)
    
    assert len(X_bags) == 3, "Should have 3 bags"
    assert len(proportions) == 3, "Should have 3 proportions"
    assert sum(len(bag) for bag in X_bags) == 100, "All samples should be in bags"
    print("  ✓ Bag creation works correctly")


def test_meanmap():
    """Test MeanMap algorithm."""
    print("Testing MeanMap algorithm...")
    X_bags, proportions, X_test, y_test = generate_llp_dataset(
        n_samples=200, n_features=10, n_bags=5, random_state=42
    )
    
    model = MeanMap(random_state=42)
    model.fit(X_bags, proportions)
    
    assert model.is_fitted_, "Model should be fitted"
    assert model.mu_pos is not None, "Should have positive class mean"
    assert model.mu_neg is not None, "Should have negative class mean"
    
    y_pred = model.predict(X_test)
    assert len(y_pred) == len(y_test), "Predictions should match test size"
    assert all(p in [0, 1] for p in y_pred), "Predictions should be binary"
    
    accuracy = np.mean(y_pred == y_test)
    assert accuracy > 0.4, f"Accuracy {accuracy} should be > 0.4 (better than random)"
    print(f"  ✓ MeanMap works correctly (accuracy: {accuracy:.3f})")


def test_propsvm():
    """Test PropSVM algorithm."""
    print("Testing PropSVM algorithm...")
    X_bags, proportions, X_test, y_test = generate_llp_dataset(
        n_samples=200, n_features=10, n_bags=5, random_state=42
    )
    
    model = PropSVM(max_iter=20, random_state=42)
    model.fit(X_bags, proportions)
    
    assert model.is_fitted_, "Model should be fitted"
    assert model.svm is not None, "Should have trained SVM"
    
    y_pred = model.predict(X_test)
    assert len(y_pred) == len(y_test), "Predictions should match test size"
    assert all(p in [0, 1] for p in y_pred), "Predictions should be binary"
    
    accuracy = np.mean(y_pred == y_test)
    assert accuracy > 0.3, f"Accuracy {accuracy} should be > 0.3"
    print(f"  ✓ PropSVM works correctly (accuracy: {accuracy:.3f})")


def test_proportion_preservation():
    """Test that learned proportions approximate true proportions."""
    print("Testing proportion preservation...")
    X_bags, true_props, X_test, y_test = generate_llp_dataset(
        n_samples=300, n_features=10, n_bags=5, random_state=42
    )
    
    model = PropSVM(max_iter=30, random_state=42)
    model.fit(X_bags, true_props)
    
    # Check learned proportions on training bags
    learned_props = []
    for bag in X_bags:
        pred = model.predict(bag)
        learned_props.append(np.mean(pred))
    
    learned_props = np.array(learned_props)
    
    # Learned proportions should be close to true proportions
    diff = np.abs(learned_props - true_props)
    mean_diff = np.mean(diff)
    
    print(f"  True proportions: {true_props.round(3)}")
    print(f"  Learned proportions: {learned_props.round(3)}")
    print(f"  Mean absolute difference: {mean_diff:.3f}")
    assert mean_diff < 0.2, f"Mean difference {mean_diff} should be < 0.2"
    print("  ✓ Proportions are approximately preserved")


if __name__ == "__main__":
    print("=" * 80)
    print("Running LLP Algorithm Tests")
    print("=" * 80)
    print()
    
    try:
        test_generate_dataset()
        test_create_bags()
        test_meanmap()
        test_propsvm()
        test_proportion_preservation()
        
        print()
        print("=" * 80)
        print("All tests passed successfully! ✓")
        print("=" * 80)
    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        raise

"""
Learning with Label Proportions (LLP) package.

This package provides implementations of various algorithms for learning
classifiers from label proportions instead of individual instance labels.
"""

__version__ = "0.1.0"

from llp.algorithms.base import BaseLLPClassifier
from llp.algorithms.propsvm import PropSVM
from llp.algorithms.meanmap import MeanMap

__all__ = [
    "BaseLLPClassifier",
    "PropSVM",
    "MeanMap",
]

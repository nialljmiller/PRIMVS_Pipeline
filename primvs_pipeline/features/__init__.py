"""
Features Module

Provides statistical feature calculation for variable star lightcurves.

This module computes a comprehensive set of time-series statistics and
periodogram-based features used for characterizing variable stars.
"""

from .statistics import FeatureCalculator, calculate_all_features
from .column_definitions import get_column_names, get_feature_columns

__all__ = [
    "FeatureCalculator",
    "calculate_all_features",
    "get_column_names",
    "get_feature_columns"
]


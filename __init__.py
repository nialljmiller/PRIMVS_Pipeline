"""
Features Module

Provides statistical feature calculation for variable star lightcurves.

This module computes a comprehensive set of time-series statistics and
periodogram-based features used for characterizing variable stars.

All statistical and periodogram functionality is backed by the stochistats
package (pip install stochistats).
"""

from .statistics import FeatureCalculator, calculate_all_features
from .column_definitions import get_column_names, get_feature_columns

# Re-export periodogram methods so the pipeline can use them directly
# via:  from primvs_pipeline.features import LS, PDM, CE, GP
try:
    from stochistats import (
        LS, PDM, CE, GP,
        extract_peaks, check_alias, exclude_alias_regions,
        make_frequency_grid,
        sine_fit, bin_lc, phaser,
    )
except ImportError:
    pass  # stochistats not installed; statistics.py will log the warning

__all__ = [
    "FeatureCalculator",
    "calculate_all_features",
    "get_column_names",
    "get_feature_columns",
    # Periodograms (from stochistats)
    "LS", "PDM", "CE", "GP",
    "extract_peaks", "check_alias", "exclude_alias_regions",
    "make_frequency_grid",
    # Fitting (from stochistats)
    "sine_fit", "bin_lc", "phaser",
]

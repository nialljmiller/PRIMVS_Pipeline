"""
Statistical Features Module

Provides calculation of time-series statistics for variable star lightcurves.

This module wraps the StochiStats library and provides a clean interface
for calculating all statistical features used in the PRIMVS catalogue.

Author: Niall Miller (refactored)
Date: 2025-10-21
Updated: 2026-02-09 — Clean stochistats integration (no more sys.path hack)
"""

import numpy as np
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Clean import from the installed stochistats package
# ---------------------------------------------------------------------------
try:
    import stochistats as ss
    from stochistats import (
        cody_M, Stetson_K, Eta, Eta_e, medianBRP,
        RangeCumSum, MaxSlope, MedianAbsDev, Meanvariance,
        PercentAmplitude, RoMS, ptop_var, lagauto, AndersonDarling,
        stdnxs, weighted_mean, weighted_variance, weighted_skew,
        weighted_kurtosis, mu, sigma, skewness, kurtosis,
    )
    # Periodogram methods
    from stochistats import (
        LS, PDM, CE, GP,
        extract_peaks, check_alias, exclude_alias_regions,
        make_frequency_grid,
    )
    # Fitting utilities
    from stochistats import sine_fit, bin_lc, phaser
    STOCHISTATS_AVAILABLE = True
    logger.debug(f"StochiStats {ss.__version__} loaded successfully")
except ImportError as e:
    logger.warning(f"StochiStats not available: {e}. "
                   "Install with: pip install -e /path/to/StochiStats")
    STOCHISTATS_AVAILABLE = False


class FeatureCalculator:
    """
    Calculator for time-series statistical features.

    Provides a clean interface to calculate all features used in PRIMVS,
    backed by the stochistats package.
    """

    def __init__(self, features_to_calculate: Optional[List[str]] = None):
        """
        Initialize feature calculator.

        Args:
            features_to_calculate: List of feature names to calculate.
                                   If None, calculates all features.
        """
        self.features_to_calculate = features_to_calculate

        if not STOCHISTATS_AVAILABLE:
            logger.error(
                "StochiStats not available — feature calculation will fail. "
                "Install with: pip install stochistats"
            )

    # ------------------------------------------------------------------
    # Basic lightcurve statistics (no stochistats dependency)
    # ------------------------------------------------------------------
    def calculate_basic_stats(
        self,
        mag: np.ndarray,
        magerr: np.ndarray,
        time: np.ndarray,
    ) -> Dict[str, float]:
        """
        Calculate basic lightcurve statistics.

        Args:
            mag: Magnitude array
            magerr: Magnitude error array
            time: Time array

        Returns:
            Dictionary of basic statistics
        """
        q1, q50, q99 = np.percentile(mag, [1, 50, 99])

        stats = {
            "mag_n": len(mag),
            "mag_avg": q50,
            "magerr_avg": np.median(magerr),
            "time_range": np.ptp(time),
            "true_amplitude": abs(q99 - q1),
        }

        return stats

    # ------------------------------------------------------------------
    # Variability statistics
    # ------------------------------------------------------------------
    def calculate_variability_stats(
        self,
        mag: np.ndarray,
        magerr: np.ndarray,
        time: np.ndarray,
    ) -> Dict[str, float]:
        """
        Calculate variability statistics via stochistats.

        Args:
            mag: Magnitude array
            magerr: Magnitude error array
            time: Time array

        Returns:
            Dictionary of variability statistics
        """
        _nan_result = {
            name: np.nan
            for name in [
                "Cody_M", "stet_k", "eta", "eta_e", "med_BRP",
                "range_cum_sum", "max_slope", "MAD", "mean_var",
                "percent_amp", "roms", "p_to_p_var", "lag_auto",
                "AD", "std_nxs",
            ]
        }

        if not STOCHISTATS_AVAILABLE:
            logger.warning("StochiStats not available, returning NaN values")
            return _nan_result

        def _safe(fn, *args):
            """Wrap a single statistic call so one failure doesn't kill all."""
            try:
                val = fn(*args)
                return float(val) if val is not None else np.nan
            except Exception as exc:
                logger.debug(f"{fn.__name__} failed: {exc}")
                return np.nan

        stats = {
            "Cody_M":         _safe(cody_M, mag, time),
            "stet_k":         _safe(Stetson_K, mag, magerr),
            "eta":            _safe(Eta, mag, time),
            "eta_e":          _safe(Eta_e, mag, time),
            "med_BRP":        _safe(medianBRP, mag, magerr),
            "range_cum_sum":  _safe(RangeCumSum, mag),
            "max_slope":      _safe(MaxSlope, mag, time),
            "MAD":            _safe(MedianAbsDev, mag, magerr),
            "mean_var":       _safe(Meanvariance, mag),
            "percent_amp":    _safe(PercentAmplitude, mag),
            "roms":           _safe(RoMS, mag, magerr),
            "p_to_p_var":     _safe(ptop_var, mag, magerr),
            "lag_auto":       _safe(lagauto, mag),
            "AD":             _safe(AndersonDarling, mag),
            "std_nxs":        _safe(stdnxs, mag, magerr),
        }

        return stats

    # ------------------------------------------------------------------
    # Statistical moments
    # ------------------------------------------------------------------
    def calculate_moments(
        self,
        mag: np.ndarray,
        magerr: np.ndarray,
    ) -> Dict[str, float]:
        """
        Calculate statistical moments (weighted and unweighted).

        Args:
            mag: Magnitude array
            magerr: Magnitude error array

        Returns:
            Dictionary of moment statistics
        """
        if not STOCHISTATS_AVAILABLE:
            logger.warning("StochiStats not available, using numpy fallbacks")
            w = 1.0 / magerr ** 2
            wmean = np.average(mag, weights=w)
            return {
                "weight_mean": wmean,
                "weight_std": np.sqrt(np.average((mag - wmean) ** 2, weights=w)),
                "weight_skew": np.nan,
                "weight_kurt": np.nan,
                "mean": np.mean(mag),
                "std": np.std(mag),
                "skew": np.nan,
                "kurt": np.nan,
            }

        def _safe(fn, *args):
            try:
                val = fn(*args)
                return float(val) if val is not None else np.nan
            except Exception as exc:
                logger.debug(f"{fn.__name__} failed: {exc}")
                return np.nan

        stats = {
            "weight_mean": _safe(weighted_mean, mag, magerr),
            "weight_std":  _safe(weighted_variance, mag, magerr),
            "weight_skew": _safe(weighted_skew, mag, magerr),
            "weight_kurt": _safe(weighted_kurtosis, mag, magerr),
            "mean":        _safe(mu, mag),
            "std":         _safe(sigma, mag),
            "skew":        _safe(skewness, mag),
            "kurt":        _safe(kurtosis, mag),
        }

        return stats

    # ------------------------------------------------------------------
    # Aggregate
    # ------------------------------------------------------------------
    def calculate_all(
        self,
        mag: np.ndarray,
        magerr: np.ndarray,
        time: np.ndarray,
    ) -> Dict[str, float]:
        """
        Calculate all statistical features.

        Args:
            mag: Magnitude array
            magerr: Magnitude error array
            time: Time array

        Returns:
            Dictionary containing all features
        """
        features = {}
        features.update(self.calculate_basic_stats(mag, magerr, time))
        features.update(self.calculate_variability_stats(mag, magerr, time))
        features.update(self.calculate_moments(mag, magerr))
        logger.debug(f"Calculated {len(features)} statistical features")
        return features


# ------------------------------------------------------------------
# Module-level convenience function
# ------------------------------------------------------------------
def calculate_all_features(
    mag: np.ndarray,
    magerr: np.ndarray,
    time: np.ndarray,
) -> Dict[str, float]:
    """
    Convenience function to calculate all features.

    Args:
        mag: Magnitude array
        magerr: Magnitude error array
        time: Time array

    Returns:
        Dictionary of all features

    Example:
        >>> features = calculate_all_features(mag, magerr, time)
        >>> print(features['Cody_M'])
    """
    calculator = FeatureCalculator()
    return calculator.calculate_all(mag, magerr, time)

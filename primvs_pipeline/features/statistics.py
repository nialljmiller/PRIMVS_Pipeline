"""
Statistical Features Module

Provides calculation of time-series statistics for variable star lightcurves.

This module wraps the StochiStats library and provides a clean interface
for calculating all statistical features used in the PRIMVS catalogue.

Author: Niall Miller (refactored)
Date: 2025-10-21
"""

import numpy as np
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

# Import StochiStats functions
# Note: These will be imported from the integrated StochiStats module
try:
    import sys
    from pathlib import Path
    # Add StochiStats to path temporarily for development
    stochi_path = Path(__file__).parent.parent.parent.parent / 'StochiStats'
    if stochi_path.exists() and str(stochi_path) not in sys.path:
        sys.path.insert(0, str(stochi_path))
    
    from StochiStats import (
        cody_M, Stetson_K, Eta, Eta_e, medianBRP,
        RangeCumSum, MaxSlope, MedianAbsDev, Meanvariance,
        PercentAmplitude, RoMS, ptop_var, lagauto, AndersonDarling,
        stdnxs, weighted_mean, weighted_variance, weighted_skew,
        weighted_kurtosis, mu, sigma, skewness, kurtosis
    )
    STOCHISTATS_AVAILABLE = True
except ImportError as e:
    logger.warning(f"StochiStats not available: {e}")
    STOCHISTATS_AVAILABLE = False


class FeatureCalculator:
    """
    Calculator for time-series statistical features.
    
    Provides a clean interface to calculate all features used in PRIMVS.
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
            logger.error("StochiStats not available - feature calculation will fail")
    
    def calculate_basic_stats(
        self,
        mag: np.ndarray,
        magerr: np.ndarray,
        time: np.ndarray
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
            'mag_n': len(mag),
            'mag_avg': q50,
            'magerr_avg': np.median(magerr),
            'time_range': np.ptp(time),
            'true_amplitude': abs(q99 - q1),
        }
        
        return stats
    
    def calculate_variability_stats(
        self,
        mag: np.ndarray,
        magerr: np.ndarray,
        time: np.ndarray
    ) -> Dict[str, float]:
        """
        Calculate variability statistics.
        
        Args:
            mag: Magnitude array
            magerr: Magnitude error array
            time: Time array
            
        Returns:
            Dictionary of variability statistics
        """
        if not STOCHISTATS_AVAILABLE:
            logger.warning("StochiStats not available, returning NaN values")
            return {name: np.nan for name in [
                'Cody_M', 'stet_k', 'eta', 'eta_e', 'med_BRP',
                'range_cum_sum', 'max_slope', 'MAD', 'mean_var',
                'percent_amp', 'roms', 'p_to_p_var', 'lag_auto',
                'AD', 'std_nxs'
            ]}
        
        try:
            stats = {
                'Cody_M': cody_M(mag, time),
                'stet_k': Stetson_K(mag, magerr),
                'eta': Eta(mag, time),
                'eta_e': Eta_e(mag, time),
                'med_BRP': medianBRP(mag, magerr),
                'range_cum_sum': RangeCumSum(mag),
                'max_slope': MaxSlope(mag, time),
                'MAD': MedianAbsDev(mag, magerr),
                'mean_var': Meanvariance(mag),
                'percent_amp': PercentAmplitude(mag),
                'roms': RoMS(mag, magerr),
                'p_to_p_var': ptop_var(mag, magerr),
                'lag_auto': lagauto(mag),
                'AD': AndersonDarling(mag),
                'std_nxs': stdnxs(mag, magerr),
            }
        except Exception as e:
            logger.error(f"Error calculating variability stats: {e}")
            stats = {name: np.nan for name in [
                'Cody_M', 'stet_k', 'eta', 'eta_e', 'med_BRP',
                'range_cum_sum', 'max_slope', 'MAD', 'mean_var',
                'percent_amp', 'roms', 'p_to_p_var', 'lag_auto',
                'AD', 'std_nxs'
            ]}
        
        return stats
    
    def calculate_moments(
        self,
        mag: np.ndarray,
        magerr: np.ndarray
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
            stats = {
                'weight_mean': np.average(mag, weights=1/magerr**2),
                'weight_std': np.sqrt(np.average((mag - np.average(mag, weights=1/magerr**2))**2, weights=1/magerr**2)),
                'weight_skew': np.nan,
                'weight_kurt': np.nan,
                'mean': np.mean(mag),
                'std': np.std(mag),
                'skew': np.nan,
                'kurt': np.nan,
            }
            return stats
        
        try:
            stats = {
                'weight_mean': weighted_mean(mag, magerr),
                'weight_std': weighted_variance(mag, magerr),
                'weight_skew': weighted_skew(mag, magerr),
                'weight_kurt': weighted_kurtosis(mag, magerr),
                'mean': mu(mag),
                'std': sigma(mag),
                'skew': skewness(mag),
                'kurt': kurtosis(mag),
            }
        except Exception as e:
            logger.error(f"Error calculating moments: {e}")
            stats = {
                'weight_mean': np.nan,
                'weight_std': np.nan,
                'weight_skew': np.nan,
                'weight_kurt': np.nan,
                'mean': np.mean(mag),
                'std': np.std(mag),
                'skew': np.nan,
                'kurt': np.nan,
            }
        
        return stats
    
    def calculate_all(
        self,
        mag: np.ndarray,
        magerr: np.ndarray,
        time: np.ndarray
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
        
        # Basic stats
        features.update(self.calculate_basic_stats(mag, magerr, time))
        
        # Variability stats
        features.update(self.calculate_variability_stats(mag, magerr, time))
        
        # Moments
        features.update(self.calculate_moments(mag, magerr))
        
        logger.debug(f"Calculated {len(features)} statistical features")
        
        return features


def calculate_all_features(
    mag: np.ndarray,
    magerr: np.ndarray,
    time: np.ndarray
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


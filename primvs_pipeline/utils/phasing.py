"""
Phase Calculation Utilities

Provides efficient phase calculation for periodic variable stars.

Author: Niall Miller
Date: 2025-10-21
"""

import numpy as np
from typing import Union


def phaser(time: Union[np.ndarray, list, float], period: float) -> np.ndarray:
    """
    Calculate phase for given times and period.
    
    Fully vectorized implementation for maximum performance.
    Phase is calculated as (time / period) mod 1, ensuring values in [0, 1).
    
    Args:
        time: Time values (MJD or any consistent time unit)
        period: Period in same units as time
        
    Returns:
        Phase values in range [0, 1)
        
    Example:
        >>> time = np.array([0, 1, 2, 3, 4])
        >>> period = 2.0
        >>> phase = phaser(time, period)
        >>> print(phase)
        [0.  0.5 0.  0.5 0. ]
        
    Notes:
        - Handles period=0 by setting period=1 to avoid division by zero
        - Fully vectorized for performance
        - Returns values in [0, 1) range
    """
    # Convert to numpy array if needed
    time = np.asarray(time)
    
    # Handle zero period (should not happen, but be safe)
    if period == 0:
        period = 1.0
    
    # Vectorized phase calculation
    phase = np.mod(time, period) / period
    
    return phase


def phase_fold(time: np.ndarray, mag: np.ndarray, period: float) -> tuple:
    """
    Phase-fold a lightcurve and sort by phase.
    
    Args:
        time: Time values
        mag: Magnitude values
        period: Period for folding
        
    Returns:
        Tuple of (sorted_phase, sorted_mag)
        
    Example:
        >>> time = np.array([0, 1, 2, 3, 4])
        >>> mag = np.array([15.0, 15.2, 15.0, 15.2, 15.0])
        >>> period = 2.0
        >>> phase, folded_mag = phase_fold(time, mag, period)
    """
    phase = phaser(time, period)
    
    # Sort by phase
    sort_idx = np.argsort(phase)
    sorted_phase = phase[sort_idx]
    sorted_mag = mag[sort_idx]
    
    return sorted_phase, sorted_mag


def phase_bins(phase: np.ndarray, mag: np.ndarray, n_bins: int = 20) -> tuple:
    """
    Bin phase-folded lightcurve data.
    
    Useful for creating smoothed phase curves or calculating phase-dependent statistics.
    
    Args:
        phase: Phase values in [0, 1)
        mag: Magnitude values
        n_bins: Number of phase bins
        
    Returns:
        Tuple of (bin_centers, bin_means, bin_stds)
        
    Example:
        >>> phase = np.random.uniform(0, 1, 100)
        >>> mag = 15.0 + 0.5 * np.sin(2 * np.pi * phase)
        >>> centers, means, stds = phase_bins(phase, mag, n_bins=10)
    """
    bins = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    bin_means = np.zeros(n_bins)
    bin_stds = np.zeros(n_bins)
    
    for i in range(n_bins):
        mask = (phase >= bins[i]) & (phase < bins[i + 1])
        if np.sum(mask) > 0:
            bin_means[i] = np.mean(mag[mask])
            bin_stds[i] = np.std(mag[mask])
        else:
            bin_means[i] = np.nan
            bin_stds[i] = np.nan
    
    return bin_centers, bin_means, bin_stds


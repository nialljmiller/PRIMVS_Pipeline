"""
Quality Filters Module

Provides quality filtering for lightcurve data.

Implements various quality cuts to remove bad photometric measurements
and ensure only reliable data is used in the analysis.

Author: Niall Miller (refactored)
Date: 2025-10-21
"""

import numpy as np
from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class QualityFilter:
    """
    Quality filter for lightcurve data.
    
    Applies configurable quality cuts to remove unreliable measurements.
    
    Attributes:
        max_chi: Maximum chi-squared value
        max_ast_res_chisq: Maximum astrometric residual chi-squared
        max_magerr_sigma: Maximum magnitude error in units of sigma
        require_positive_mag: Require positive magnitudes
        require_positive_magerr: Require positive magnitude errors
    """
    
    def __init__(
        self,
        max_chi: float = 10.0,
        max_ast_res_chisq: float = 20.0,
        max_magerr_sigma: float = 4.0,
        require_positive_mag: bool = True,
        require_positive_magerr: bool = True
    ):
        """
        Initialize quality filter.
        
        Args:
            max_chi: Maximum chi-squared value
            max_ast_res_chisq: Maximum astrometric residual chi-squared
            max_magerr_sigma: Maximum magnitude error (in units of sigma)
            require_positive_mag: Require mag > 0
            require_positive_magerr: Require magerr > 0
        """
        self.max_chi = max_chi
        self.max_ast_res_chisq = max_ast_res_chisq
        self.max_magerr_sigma = max_magerr_sigma
        self.require_positive_mag = require_positive_mag
        self.require_positive_magerr = require_positive_magerr
        
        logger.debug(f"Initialized QualityFilter with chi<{max_chi}, ast_chi<{max_ast_res_chisq}")
    
    def apply(
        self,
        lightcurve: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """
        Apply quality filters to lightcurve.
        
        Args:
            lightcurve: Dictionary containing lightcurve arrays
            
        Returns:
            Filtered lightcurve dictionary
            
        Example:
            >>> filter = QualityFilter(max_chi=10.0)
            >>> filtered_lc = filter.apply(lightcurve)
        """
        n_initial = len(lightcurve['mag'])
        
        # Build filter mask
        mask = np.ones(n_initial, dtype=bool)
        
        # Filter by chi-squared
        if 'chi' in lightcurve:
            chi_mask = lightcurve['chi'] < self.max_chi
            mask &= chi_mask
            logger.debug(f"Chi filter: {np.sum(chi_mask)}/{n_initial} points pass")
        
        # Filter by astrometric residual chi-squared
        if 'ast_res_chisq' in lightcurve:
            ast_mask = lightcurve['ast_res_chisq'] < self.max_ast_res_chisq
            mask &= ast_mask
            logger.debug(f"Astrometric chi filter: {np.sum(ast_mask)}/{n_initial} points pass")
        
        # Filter by positive magnitudes
        if self.require_positive_mag:
            mag_mask = lightcurve['mag'] > 0
            mask &= mag_mask
            logger.debug(f"Positive mag filter: {np.sum(mag_mask)}/{n_initial} points pass")
        
        # Filter by positive magnitude errors
        if self.require_positive_magerr:
            magerr_mask = lightcurve['magerr'] > 0
            mask &= magerr_mask
            logger.debug(f"Positive magerr filter: {np.sum(magerr_mask)}/{n_initial} points pass")
        
        # Filter by magnitude error sigma clipping
        if self.max_magerr_sigma > 0:
            magerr_sigma = np.std(lightcurve['magerr'])
            magerr_threshold = self.max_magerr_sigma * magerr_sigma
            sigma_mask = lightcurve['magerr'] <= magerr_threshold
            mask &= sigma_mask
            logger.debug(f"Magerr sigma filter: {np.sum(sigma_mask)}/{n_initial} points pass")
        
        # Apply mask to all arrays
        filtered_lc = {}
        for key, value in lightcurve.items():
            if isinstance(value, np.ndarray) and len(value) == n_initial:
                filtered_lc[key] = value[mask]
            else:
                filtered_lc[key] = value
        
        n_final = len(filtered_lc['mag'])
        logger.info(f"Quality filtering: {n_initial} -> {n_final} points ({n_final/n_initial*100:.1f}%)")
        
        return filtered_lc


def apply_quality_filters(
    mag: np.ndarray,
    magerr: np.ndarray,
    time: np.ndarray,
    chi: Optional[np.ndarray] = None,
    ast_res_chisq: Optional[np.ndarray] = None,
    max_chi: float = 10.0,
    max_ast_res_chisq: float = 20.0,
    max_magerr_sigma: float = 4.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convenience function to apply quality filters to arrays.
    
    Args:
        mag: Magnitude array
        magerr: Magnitude error array
        time: Time array
        chi: Chi-squared array (optional)
        ast_res_chisq: Astrometric residual chi-squared array (optional)
        max_chi: Maximum chi-squared value
        max_ast_res_chisq: Maximum astrometric residual chi-squared
        max_magerr_sigma: Maximum magnitude error in units of sigma
        
    Returns:
        Tuple of filtered (mag, magerr, time) arrays
        
    Example:
        >>> mag_f, magerr_f, time_f = apply_quality_filters(
        ...     mag, magerr, time, chi, ast_res_chisq
        ... )
    """
    # Build lightcurve dictionary
    lightcurve = {
        'mag': mag,
        'magerr': magerr,
        'time': time
    }
    
    if chi is not None:
        lightcurve['chi'] = chi
    
    if ast_res_chisq is not None:
        lightcurve['ast_res_chisq'] = ast_res_chisq
    
    # Apply filters
    filter = QualityFilter(
        max_chi=max_chi,
        max_ast_res_chisq=max_ast_res_chisq,
        max_magerr_sigma=max_magerr_sigma
    )
    
    filtered_lc = filter.apply(lightcurve)
    
    return filtered_lc['mag'], filtered_lc['magerr'], filtered_lc['time']


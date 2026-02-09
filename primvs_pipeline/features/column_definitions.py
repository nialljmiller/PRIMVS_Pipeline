"""
Column Definitions Module

Defines the standard column names and structure for the PRIMVS catalogue.

This centralizes all column definitions to ensure consistency across the pipeline.

Author: Niall Miller (refactored)
Date: 2025-10-21
"""

from typing import List, Dict


# Basic lightcurve statistics
BASIC_STATS_COLUMNS = [
    'mag_n',           # Number of observations
    'mag_avg',         # Average magnitude
    'magerr_avg',      # Average magnitude error
    'time_range',      # Time baseline
]

# Variability statistics
VARIABILITY_COLUMNS = [
    'Cody_M',          # Cody M statistic
    'stet_k',          # Stetson K statistic
    'eta',             # Eta variability index
    'eta_e',           # Eta_e variability index
    'med_BRP',         # Median buffer range percentage
    'range_cum_sum',   # Range of cumulative sum
    'max_slope',       # Maximum slope
    'MAD',             # Median absolute deviation
    'mean_var',        # Mean variance
    'percent_amp',     # Percent amplitude
    'true_amplitude',  # True amplitude (99th - 1st percentile)
    'roms',            # RoMS statistic
    'p_to_p_var',      # Point-to-point variance
    'lag_auto',        # Lag-1 autocorrelation
    'AD',              # Anderson-Darling statistic
    'std_nxs',         # Normalized excess variance
]

# Weighted moments
WEIGHTED_MOMENTS_COLUMNS = [
    'weight_mean',     # Weighted mean
    'weight_std',      # Weighted standard deviation
    'weight_skew',     # Weighted skewness
    'weight_kurt',     # Weighted kurtosis
]

# Basic moments
BASIC_MOMENTS_COLUMNS = [
    'mean',            # Mean
    'std',             # Standard deviation
    'skew',            # Skewness
    'kurt',            # Kurtosis
]

# Period-finding results (per method: LS, PDM, CE, GP)
def get_periodogram_columns(method: str) -> List[str]:
    """
    Get column names for a specific periodogram method.
    
    Args:
        method: Method name ('ls', 'pdm', 'ce', 'gp')
        
    Returns:
        List of column names for this method
    """
    prefix = method.lower()
    
    if method.lower() == 'gp':
        # Gaussian Process has different columns
        return [
            f'{prefix}_lnlike',    # Log likelihood
            f'{prefix}_b',         # Bandwidth parameter
            f'{prefix}_c',         # Scale parameter
            f'{prefix}_p',         # Period
            f'{prefix}_fap',       # False alarm probability
            f'Cody_Q_{prefix}',    # Cody Q statistic
        ]
    else:
        # Standard periodogram columns
        return [
            f'{prefix}_p',             # Best period
            f'{prefix}_y_y_0',         # Primary peak power
            f'{prefix}_peak_width_0',  # Primary peak width
            f'{prefix}_period1',       # Secondary period
            f'{prefix}_y_y_1',         # Secondary peak power
            f'{prefix}_peak_width_1',  # Secondary peak width
            f'{prefix}_period2',       # Tertiary period
            f'{prefix}_y_y_2',         # Tertiary peak power
            f'{prefix}_peak_width_2',  # Tertiary peak width
            f'{prefix}_q001',          # 0.1% quantile
            f'{prefix}_q01',           # 1% quantile
            f'{prefix}_q1',            # 10% quantile
            f'{prefix}_q25',           # 25% quantile
            f'{prefix}_q50',           # 50% quantile (median)
            f'{prefix}_q75',           # 75% quantile
            f'{prefix}_q99',           # 99% quantile
            f'{prefix}_q999',          # 99.9% quantile
            f'{prefix}_q9999',         # 99.99% quantile
            f'{prefix}_fap',           # False alarm probability
            f'{prefix}_bal_fap',       # Balanced FAP (LS only)
            f'Cody_Q_{prefix}',        # Cody Q statistic
        ]


# Best period/FAP columns
BEST_PERIOD_COLUMNS = [
    'true_period',     # Best period across all methods
    'best_fap',        # Best (minimum) FAP
    'best_method',     # Method that gave best FAP
    'trans_flag',      # Transit detection flag
]

# Metadata columns (from VIRAC)
METADATA_COLUMNS = [
    'sourceid',        # VIRAC source ID
    'ra',              # Right ascension
    'ra_error',        # RA error
    'dec',             # Declination
    'dec_error',       # Dec error
    'l',               # Galactic longitude
    'b',               # Galactic latitude
    'parallax',        # Parallax
    'parallax_error',  # Parallax error
    'pmra',            # Proper motion in RA
    'pmra_error',      # PM RA error
    'pmdec',           # Proper motion in Dec
    'pmdec_error',     # PM Dec error
    'chisq',           # Astrometric fit chi-squared
    'uwe',             # Unit weight error
]

# Multi-band photometry columns
def get_photometry_columns(band: str) -> List[str]:
    """
    Get photometry column names for a specific band.
    
    Args:
        band: Band name ('ks', 'z', 'y', 'j', 'h')
        
    Returns:
        List of column names for this band
    """
    prefix = band.lower()
    return [
        f'{prefix}_n_detections',
        f'{prefix}_n_observations',
        f'{prefix}_n_ambiguous',
        f'{prefix}_n_chilt5',
        f'{prefix}_med_mag',
        f'{prefix}_mean_mag',
        f'{prefix}_ivw_mean_mag',
        f'{prefix}_chilt5_ivw_mean_mag',
        f'{prefix}_std_mag',
        f'{prefix}_mad_mag',
        f'{prefix}_ivw_err_mag',
        f'{prefix}_chilt5_ivw_err_mag',
    ]


def get_column_names() -> List[str]:
    """
    Get complete list of column names for PRIMVS catalogue.
    
    Returns:
        List of all column names in order
    """
    columns = []
    
    # Metadata
    columns.extend(METADATA_COLUMNS)
    
    # Multi-band photometry
    for band in ['ks', 'z', 'y', 'j', 'h']:
        columns.extend(get_photometry_columns(band))
    
    # Basic statistics
    columns.extend(BASIC_STATS_COLUMNS)
    
    # Variability statistics
    columns.extend(VARIABILITY_COLUMNS)
    
    # Moments
    columns.extend(WEIGHTED_MOMENTS_COLUMNS)
    columns.extend(BASIC_MOMENTS_COLUMNS)
    
    # Periodogram results
    for method in ['ls', 'pdm', 'ce', 'gp']:
        columns.extend(get_periodogram_columns(method))
    
    # Best period/FAP
    columns.extend(BEST_PERIOD_COLUMNS)
    
    return columns


def get_feature_columns() -> List[str]:
    """
    Get list of feature columns (excluding metadata).
    
    Returns:
        List of feature column names
    """
    columns = []
    
    columns.extend(BASIC_STATS_COLUMNS)
    columns.extend(VARIABILITY_COLUMNS)
    columns.extend(WEIGHTED_MOMENTS_COLUMNS)
    columns.extend(BASIC_MOMENTS_COLUMNS)
    
    for method in ['ls', 'pdm', 'ce', 'gp']:
        columns.extend(get_periodogram_columns(method))
    
    columns.extend(BEST_PERIOD_COLUMNS)
    
    return columns


def get_column_descriptions() -> Dict[str, str]:
    """
    Get descriptions for all columns.
    
    Returns:
        Dictionary mapping column names to descriptions
    """
    descriptions = {
        # Basic stats
        'mag_n': 'Number of observations',
        'mag_avg': 'Average magnitude',
        'magerr_avg': 'Average magnitude error',
        'time_range': 'Time baseline (days)',
        
        # Variability
        'Cody_M': 'Cody M statistic',
        'stet_k': 'Stetson K statistic',
        'eta': 'Eta variability index',
        'eta_e': 'Eta_e variability index',
        'true_amplitude': 'True amplitude (99th - 1st percentile)',
        
        # Period
        'true_period': 'Best period (days)',
        'best_fap': 'Best false alarm probability',
        'best_method': 'Method that gave best period',
        
        # Metadata
        'sourceid': 'VIRAC source identifier',
        'ra': 'Right ascension (degrees)',
        'dec': 'Declination (degrees)',
        'l': 'Galactic longitude (degrees)',
        'b': 'Galactic latitude (degrees)',
    }
    
    return descriptions


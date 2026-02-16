"""
VIRAC Database Interface

Provides clean access to VIRAC lightcurve data extracted by VIRAC_extract.

Data format: CSV files in hierarchical directory structure:
    {lc_dir}/{first3digits}/{next3digits}/{sourceid}.csv

CSV columns: mjd, ks_mag, ks_err, z_mag, z_err, y_mag, y_err, j_mag, j_err,
             h_mag, h_err, seeing, exptime, skylevel, ellipticity, chi,
             ast_res_chisq, detected, filter

Author: Niall Miller (refactored)
Date: 2025-10-21
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

# Column name mapping: filter_band -> (mag_col, err_col)
FILTER_COLUMNS = {
    'Ks': ('ks_mag', 'ks_err'),
    'Z':  ('z_mag',  'z_err'),
    'Y':  ('y_mag',  'y_err'),
    'J':  ('j_mag',  'j_err'),
    'H':  ('h_mag',  'h_err'),
}


class ViracInterface:
    """
    Interface to VIRAC lightcurve data extracted by VIRAC_extract.
    
    Reads CSV files from hierarchical directory structure produced by
    virac_lightcurve_extractor.py.
    
    Attributes:
        lc_dir: Directory containing lightcurve CSV files
    """
    
    def __init__(self, lc_dir: str, meta_dir: Optional[str] = None):
        """
        Initialize VIRAC interface.
        
        Args:
            lc_dir: Path to directory containing lightcurve CSV files
            meta_dir: Unused, kept for backward compatibility
        """
        self.lc_dir = Path(lc_dir)
        
        if not self.lc_dir.exists():
            logger.warning(f"Lightcurve directory does not exist: {self.lc_dir}")
        
        logger.info(f"Initialized VIRAC interface with LC dir: {self.lc_dir}")
    
    def _resolve_path(self, source_id) -> Path:
        """
        Resolve the hierarchical path for a given source ID.
        
        Path structure: {lc_dir}/{first3}/{next3}/{sourceid}.csv
        
        Args:
            source_id: VIRAC source identifier
            
        Returns:
            Path to the CSV file
        """
        source_str = str(int(source_id))
        subdir1 = source_str[:3]
        subdir2 = source_str[3:6]
        return self.lc_dir / subdir1 / subdir2 / f"{source_str}.csv"
    
    def get_lightcurve(self, source_id: int, filter_band: str = 'Ks') -> Dict[str, np.ndarray]:
        """
        Retrieve lightcurve for a given source ID.
        
        Args:
            source_id: VIRAC source identifier
            filter_band: Filter band to extract (default: 'Ks')
            
        Returns:
            Dictionary containing lightcurve data with keys:
                - mag: Magnitude values
                - magerr: Magnitude errors
                - time: Time values (MJD)
                - chi: Chi-squared values
                - ast_res_chisq: Astrometric residual chi-squared
                - seeing: Seeing values
                - exptime: Exposure times
                - skylevel: Sky background levels
                - ellipticity: PSF ellipticity
                - detected: Detection flag
                - filter: Filter band
                - sourceid: Source ID (array)
                
        Raises:
            FileNotFoundError: If CSV file for source not found
            ValueError: If no data found for specified filter
        """
        csv_path = self._resolve_path(source_id)
        
        if not csv_path.exists():
            raise FileNotFoundError(f"Lightcurve file not found: {csv_path}")
        
        try:
            df = pd.read_csv(csv_path)
            
            # Extract data for specified filter
            lightcurve = self._extract_filter_data(df, filter_band, source_id)
            
            if len(lightcurve['mag']) == 0:
                raise ValueError(f"No data found for filter {filter_band}")
            
            logger.debug(f"Loaded lightcurve for source {source_id}: {len(lightcurve['mag'])} points")
            return lightcurve
            
        except (FileNotFoundError, ValueError):
            raise
        except Exception as e:
            logger.error(f"Error loading lightcurve for source {source_id}: {e}")
            raise
    
    def _extract_filter_data(self, df: pd.DataFrame, filter_band: str, source_id: int) -> Dict[str, np.ndarray]:
        """
        Extract data for a specific filter band from VIRAC CSV data.
        
        Args:
            df: DataFrame from CSV file
            filter_band: Filter band to extract
            source_id: Source ID for metadata
            
        Returns:
            Dictionary of extracted arrays
        """
        if filter_band not in FILTER_COLUMNS:
            raise ValueError(f"Unknown filter band: {filter_band}. Valid: {list(FILTER_COLUMNS.keys())}")
        
        mag_col, err_col = FILTER_COLUMNS[filter_band]
        
        # Filter by the 'filter' column AND require non-null magnitude
        filter_mask = (df['filter'] == filter_band) & df[mag_col].notna()
        filtered = df[filter_mask]
        
        # Build lightcurve dictionary matching the interface the pipeline expects
        lightcurve = {
            'mag': filtered[mag_col].values.astype(np.float64),
            'magerr': filtered[err_col].values.astype(np.float64),
            'time': filtered['mjd'].values.astype(np.float64),
            'chi': filtered['chi'].values.astype(np.float64),
            'ast_res_chisq': filtered['ast_res_chisq'].values.astype(np.float64),
            'seeing': filtered['seeing'].values.astype(np.float64),
            'exptime': filtered['exptime'].values.astype(np.float64),
            'skylevel': filtered['skylevel'].values.astype(np.float64),
            'ellipticity': filtered['ellipticity'].values.astype(np.float64),
            'detected': filtered['detected'].values.astype(np.int32) if 'detected' in filtered.columns else np.ones(len(filtered), dtype=np.int32),
            'filter': np.array([filter_band] * len(filtered)),
            'sourceid': np.array([source_id] * len(filtered)),
        }
        
        # Keep backward-compatible aliases
        lightcurve['mjdobs'] = lightcurve['time']
        lightcurve['hfad_mag'] = lightcurve['mag']
        lightcurve['hfad_emag'] = lightcurve['magerr']
        
        return lightcurve


def load_lightcurve(
    source_id: int,
    lc_dir: str,
    filter_band: str = 'Ks'
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convenience function to load lightcurve data.
    
    Args:
        source_id: VIRAC source identifier
        lc_dir: Directory containing lightcurve CSV files
        filter_band: Filter band to extract
        
    Returns:
        Tuple of (mag, magerr, time) arrays
        
    Example:
        >>> mag, magerr, time = load_lightcurve(10003313000040, '/path/to/virac/lcs')
    """
    interface = ViracInterface(lc_dir)
    lc = interface.get_lightcurve(source_id, filter_band)
    return lc['mag'], lc['magerr'], lc['time']
